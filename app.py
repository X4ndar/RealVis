#!/usr/bin/env python3
"""
FastAPI REST API for Stable Diffusion XL Image Generation
Based on the existing generate_image.py script with GPU optimizations.
"""

import logging
import base64
import io
import torch
import threading
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from PIL import Image
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variable to store the pipeline
pipeline = None

class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024

class GenerateResponse(BaseModel):
    image: str
    prompt: str
    generation_time: float

def check_gpu_availability():
    """Check if CUDA is available and print GPU info."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires GPU access.")
        return False
    
    logger.info(f"CUDA is available")
    logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return True

def print_gpu_usage(stage=""):
    """Print detailed GPU memory usage information."""
    if not torch.cuda.is_available():
        return
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free_memory = total_memory - reserved
    
    logger.info(f"GPU Memory Usage {stage}:")
    logger.info(f"   Total Memory: {total_memory / 1e9:.2f} GB")
    logger.info(f"   Allocated: {allocated / 1e9:.2f} GB ({allocated/total_memory*100:.1f}%)")
    logger.info(f"   Reserved: {reserved / 1e9:.2f} GB ({reserved/total_memory*100:.1f}%)")
    logger.info(f"   Free: {free_memory / 1e9:.2f} GB ({free_memory/total_memory*100:.1f}%)")

def load_optimized_model():
    """Load the Stable Diffusion XL model with optimized settings (from generate_image.py)."""
    logger.info("Loading Stable Diffusion XL model with optimizations...")
    logger.info("Model: stabilityai/stable-diffusion-xl-base-1.0")
    
    try:
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        logger.info("Moving model to GPU...")
        pipe = pipe.to("cuda")
        
        # Enable memory efficient attention for older PyTorch versions
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except:
                logger.info("xformers not available, using default attention")
        
        # Enable memory optimizations for low VRAM GPUs
        pipe.enable_model_cpu_offload()
        logger.info("Enabled model CPU offload for memory optimization")
        
        # Enable memory efficient attention
        try:
            pipe.enable_attention_slicing()
            logger.info("Enabled attention slicing")
        except:
            logger.info("Attention slicing not available")
            
        # Enable VAE slicing for lower memory usage
        try:
            pipe.enable_vae_slicing()
            logger.info("Enabled VAE slicing")
        except:
            logger.info("VAE slicing not available")
        
        logger.info("Model loaded successfully with memory optimizations!")
        print_gpu_usage("(After Model Loading)")
        
        return pipe
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Create FastAPI app
app = FastAPI(
    title="Stable Diffusion XL API",
    description="Image generation API using Stable Diffusion XL with GPU optimizations",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load the Stable Diffusion XL pipeline at startup."""
    global pipeline
    
    # Check GPU availability
    if not check_gpu_availability():
        raise RuntimeError("CUDA is not available. This app requires a GPU.")
    
    # Initial GPU usage baseline
    print_gpu_usage("(Initial Baseline)")
    
    # Load model with optimizations
    pipeline = load_optimized_model()
    if pipeline is None:
        raise RuntimeError("Failed to load Stable Diffusion XL pipeline")

@app.get("/health")
async def health_check():
    """Health check endpoint with GPU info."""
    gpu_info = {}
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(total_memory / 1e9, 2),
            "allocated_memory_gb": round(allocated / 1e9, 2),
            "reserved_memory_gb": round(reserved / 1e9, 2),
            "memory_usage_percent": round((reserved / total_memory) * 100, 1)
        }
    
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": gpu_info
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest) -> GenerateResponse:
    """Generate an image from a text prompt using optimized pipeline."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check server logs."
        )
    
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty."
        )
    
    logger.info(f"Generating image from prompt: '{prompt}'")
    logger.info(f"Parameters: steps={request.num_inference_steps}, guidance={request.guidance_scale}, size={request.width}x{request.height}")
    
    # Check GPU usage before generation
    print_gpu_usage("(Before Image Generation)")
    
    try:
        start_time = time.time()
        
        # Generate image with optimization parameters (from generate_image.py approach)
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                height=request.height,
                width=request.width
            )
        
        generated_image = result.images[0]
        generation_time = time.time() - start_time
        
        # Check GPU usage after generation
        print_gpu_usage("(After Image Generation)")
        
        # Convert PIL Image to base64 string
        img_buffer = io.BytesIO()
        generated_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        
        logger.info(f"Successfully generated image for prompt: '{prompt}' in {generation_time:.2f}s")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return GenerateResponse(
            image=img_base64,
            prompt=prompt,
            generation_time=round(generation_time, 2)
        )
        
    except Exception as e:
        error_msg = f"Failed to generate image: {str(e)}"
        logger.error(error_msg)
        
        # Clear GPU cache on error
        torch.cuda.empty_cache()
        print_gpu_usage("(After Generation Error)")
        
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Web UI for testing image generation."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎨 AI Image Generator</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
            }
            input, select, button {
                width: 100%;
                padding: 12px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                box-sizing: border-box;
            }
            input, select {
                background: rgba(255, 255, 255, 0.9);
                color: #333;
            }
            button {
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s;
                margin-top: 10px;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            }
            button:disabled {
                background: #666;
                cursor: not-allowed;
                transform: none;
            }
            .advanced {
                display: none;
                margin-top: 20px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
            }
            .advanced.show {
                display: block;
            }
            .toggle-advanced {
                background: rgba(255, 255, 255, 0.2);
                color: white;
                margin-bottom: 20px;
            }
            #result {
                margin-top: 30px;
                text-align: center;
            }
            #generatedImage {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                margin: 20px 0;
            }
            .status {
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }
            .status.loading {
                background: rgba(255, 193, 7, 0.3);
                border: 2px solid #ffc107;
            }
            .status.success {
                background: rgba(40, 167, 69, 0.3);
                border: 2px solid #28a745;
            }
            .status.error {
                background: rgba(220, 53, 69, 0.3);
                border: 2px solid #dc3545;
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }
            .api-info {
                background: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 AI Image Generator</h1>
            <p style="text-align: center; opacity: 0.9;">Stable Diffusion XL with GPU Optimizations</p>
            
            <form id="generateForm">
                <div class="form-group">
                    <label for="prompt">✨ Enter your prompt:</label>
                    <input 
                        type="text" 
                        id="prompt" 
                        placeholder="e.g., a majestic dragon flying over a cyberpunk city at sunset"
                        required
                    >
                </div>
                
                <button type="button" class="toggle-advanced" onclick="toggleAdvanced()">
                    ⚙️ Advanced Settings
                </button>
                
                <div id="advanced" class="advanced">
                    <div class="grid">
                        <div class="form-group">
                            <label for="steps">Inference Steps:</label>
                            <select id="steps">
                                <option value="15">15 (Fast)</option>
                                <option value="20" selected>20 (Balanced)</option>
                                <option value="30">30 (Quality)</option>
                                <option value="50">50 (High Quality)</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="guidance">Guidance Scale:</label>
                            <select id="guidance">
                                <option value="5.0">5.0 (Creative)</option>
                                <option value="7.5" selected>7.5 (Balanced)</option>
                                <option value="10.0">10.0 (Strict)</option>
                                <option value="15.0">15.0 (Very Strict)</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="width">Width:</label>
                            <select id="width">
                                <option value="512">512px</option>
                                <option value="768">768px</option>
                                <option value="1024" selected>1024px</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="height">Height:</label>
                            <select id="height">
                                <option value="512">512px</option>
                                <option value="768">768px</option>
                                <option value="1024" selected>1024px</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <button type="submit" id="generateBtn">
                    🚀 Generate Image
                </button>
            </form>
            
            <div id="result"></div>
            
            <div class="api-info">
                <strong>📊 API Endpoints:</strong><br>
                • <a href="/health" style="color: #ffc107;">/health</a> - Check server status<br>
                • <a href="/docs" style="color: #ffc107;">/docs</a> - API documentation<br>
                • POST /generate - Generate images
            </div>
        </div>

        <script>
            let isGenerating = false;

            function toggleAdvanced() {
                const advanced = document.getElementById('advanced');
                advanced.classList.toggle('show');
            }

            document.getElementById('generateForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                if (isGenerating) return;
                
                const prompt = document.getElementById('prompt').value.trim();
                if (!prompt) {
                    showStatus('Please enter a prompt', 'error');
                    return;
                }
                
                isGenerating = true;
                const generateBtn = document.getElementById('generateBtn');
                generateBtn.disabled = true;
                generateBtn.textContent = '⏳ Generating...';
                
                showStatus('🎨 Generating your image... This may take 20-60 seconds.', 'loading');
                
                const requestBody = {
                    prompt: prompt,
                    num_inference_steps: parseInt(document.getElementById('steps').value),
                    guidance_scale: parseFloat(document.getElementById('guidance').value),
                    width: parseInt(document.getElementById('width').value),
                    height: parseInt(document.getElementById('height').value)
                };
                
                try {
                    const startTime = Date.now();
                    
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestBody)
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Generation failed');
                    }
                    
                    const data = await response.json();
                    const totalTime = Date.now() - startTime;
                    
                    showResult(data, totalTime);
                    showStatus(`✅ Image generated successfully in ${data.generation_time}s`, 'success');
                    
                } catch (error) {
                    console.error('Error:', error);
                    showStatus(`❌ Error: ${error.message}`, 'error');
                } finally {
                    isGenerating = false;
                    generateBtn.disabled = false;
                    generateBtn.textContent = '🚀 Generate Image';
                }
            });
            
            function showStatus(message, type) {
                const result = document.getElementById('result');
                const statusDiv = document.createElement('div');
                statusDiv.className = `status ${type}`;
                statusDiv.textContent = message;
                
                // Remove previous status
                const existingStatus = result.querySelector('.status');
                if (existingStatus) {
                    existingStatus.remove();
                }
                
                result.insertBefore(statusDiv, result.firstChild);
            }
            
            function showResult(data, totalTime) {
                const result = document.getElementById('result');
                
                // Remove previous image
                const existingImage = result.querySelector('#generatedImage');
                if (existingImage) {
                    existingImage.remove();
                }
                
                const img = document.createElement('img');
                img.id = 'generatedImage';
                img.src = `data:image/png;base64,${data.image}`;
                img.alt = 'Generated Image';
                
                const info = document.createElement('div');
                info.innerHTML = `
                    <strong>📝 Prompt:</strong> ${data.prompt}<br>
                    <strong>⏱️ Generation Time:</strong> ${data.generation_time}s<br>
                    <strong>🌐 Total Request Time:</strong> ${(totalTime/1000).toFixed(2)}s
                `;
                info.style.marginTop = '15px';
                info.style.background = 'rgba(255, 255, 255, 0.1)';
                info.style.padding = '15px';
                info.style.borderRadius = '8px';
                
                result.appendChild(img);
                result.appendChild(info);
            }
            
            // Check API health on page load
            fetch('/health')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'healthy' && data.model_loaded) {
                        showStatus('🟢 Server is healthy and model is loaded', 'success');
                    } else {
                        showStatus('🟡 Server responding but model may not be loaded', 'error');
                    }
                })
                .catch(error => {
                    showStatus('🔴 Cannot connect to server', 'error');
                });
        </script>
    </body>
    </html>
    """

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "Stable Diffusion XL API with GPU Optimizations",
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /generate - Generate image from prompt",
            "health": "GET /health - Health check with GPU info",
            "docs": "GET /docs - API documentation",
            "web": "GET / - Web UI interface"
        },
        "optimizations": [
            "xformers memory efficient attention",
            "model CPU offload",
            "attention slicing",
            "VAE slicing",
            "GPU cache clearing"
        ]
    }

if __name__ == "__main__":
    # Run the app with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    ) 