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
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import DiffusionPipeline, AutoPipelineForInpainting
from PIL import Image
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables to store the pipelines
pipeline = None
inpaint_pipeline = None

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

class InpaintRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 20
    guidance_scale: float = 8.0
    strength: float = 0.99

class InpaintResponse(BaseModel):
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

def load_inpainting_model():
    """Load the SD 2.0 inpainting model with optimized settings."""
    logger.info("Loading Stable Diffusion 2.0 Inpainting model...")
    logger.info("Model: stabilityai/stable-diffusion-2-inpainting")
    
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Apply same optimizations as text-to-image model
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers for inpainting")
            except:
                logger.info("xformers not available for inpainting")
        
        pipe.enable_model_cpu_offload()
        logger.info("Enabled model CPU offload for inpainting")
        
        try:
            pipe.enable_attention_slicing()
            logger.info("Enabled attention slicing for inpainting")
        except:
            logger.info("Attention slicing not available for inpainting")
            
        try:
            pipe.enable_vae_slicing()
            logger.info("Enabled VAE slicing for inpainting")
        except:
            logger.info("VAE slicing not available for inpainting")
        
        logger.info("Inpainting model loaded successfully!")
        print_gpu_usage("(After Inpainting Model Loading)")
        
        return pipe
        
    except Exception as e:
        logger.error(f"Error loading inpainting model: {e}")
        return None

# Create FastAPI app
app = FastAPI(
    title="Stable Diffusion XL API",
    description="Image generation API using Stable Diffusion XL with GPU optimizations",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load both text-to-image and inpainting pipelines at startup."""
    global pipeline, inpaint_pipeline
    
    # Check GPU availability
    if not check_gpu_availability():
        raise RuntimeError("CUDA is not available. This app requires a GPU.")
    
    # Initial GPU usage baseline
    print_gpu_usage("(Initial Baseline)")
    
    # Load text-to-image model
    pipeline = load_optimized_model()
    if pipeline is None:
        raise RuntimeError("Failed to load Stable Diffusion XL pipeline")
    
    # Load inpainting model
    inpaint_pipeline = load_inpainting_model()
    if inpaint_pipeline is None:
        raise RuntimeError("Failed to load SD 2.0 inpainting pipeline")

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
        "text_to_image_loaded": pipeline is not None,
        "inpainting_loaded": inpaint_pipeline is not None,
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

@app.post("/inpaint", response_model=InpaintResponse)
async def inpaint_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    num_inference_steps: int = Form(20),
    guidance_scale: float = Form(8.0),
    strength: float = Form(0.99)
):
    """Inpaint an image using SDXL inpainting model."""
    global inpaint_pipeline
    
    if inpaint_pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Inpainting model not loaded. Please check server logs."
        )
    
    if not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty."
        )
    
    logger.info(f"Inpainting with prompt: '{prompt}'")
    logger.info(f"Parameters: steps={num_inference_steps}, guidance={guidance_scale}, strength={strength}")
    
    try:
        start_time = time.time()
        
        # Load and process images
        image_pil = Image.open(io.BytesIO(await image.read())).convert("RGB")
        mask_pil = Image.open(io.BytesIO(await mask.read())).convert("RGB")
        
        # Resize to 512x512 (SD 2.0 native resolution)
        image_pil = image_pil.resize((512, 512))
        mask_pil = mask_pil.resize((512, 512))
        
        print_gpu_usage("(Before Inpainting)")
        
        # Generate inpainted image
        with torch.inference_mode():
            generator = torch.Generator(device="cuda").manual_seed(int(time.time()))
            
            result = inpaint_pipeline(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator
            )
        
        inpainted_image = result.images[0]
        generation_time = time.time() - start_time
        
        print_gpu_usage("(After Inpainting)")
        
        # Convert to base64
        img_buffer = io.BytesIO()
        inpainted_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        
        logger.info(f"Successfully inpainted image in {generation_time:.2f}s")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return InpaintResponse(
            image=img_base64,
            prompt=prompt,
            generation_time=round(generation_time, 2)
        )
        
    except Exception as e:
        error_msg = f"Failed to inpaint image: {str(e)}"
        logger.error(error_msg)
        torch.cuda.empty_cache()
        print_gpu_usage("(After Inpainting Error)")
        
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Web UI for both text-to-image generation and inpainting."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎨 AI Image Studio</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
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
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }
            .subtitle {
                text-align: center;
                opacity: 0.9;
                margin-bottom: 30px;
            }
            
            /* Tab System */
            .tabs {
                display: flex;
                margin-bottom: 30px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 5px;
            }
            .tab {
                flex: 1;
                padding: 15px;
                text-align: center;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
                font-weight: bold;
            }
            .tab.active {
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
            }
            .tab:not(.active):hover {
                background: rgba(255, 255, 255, 0.1);
            }
            
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            
            /* Common Styles */
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
            }
            input, select, button, textarea {
                width: 100%;
                padding: 12px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                box-sizing: border-box;
            }
            input, select, textarea {
                background: rgba(255, 255, 255, 0.9);
                color: #333;
            }
            input[type="file"] {
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
            button.secondary {
                background: rgba(255, 255, 255, 0.2);
            }
            button.active {
                background: linear-gradient(45deg, #28a745, #20c997);
            }
            
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }
            .button-group {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }
            
            /* Advanced Settings */
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
            
                         /* Inpainting Specific */
             .workspace {
                 display: grid;
                 grid-template-columns: 1fr;
                 gap: 20px;
                 margin: 20px 0;
                 justify-items: center;
             }
            .canvas-container {
                background: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                padding: 15px;
                text-align: center;
            }
            .canvas-container h3 {
                color: #333;
                margin: 0 0 15px 0;
            }
                         canvas {
                 border: 2px solid #ddd;
                 border-radius: 5px;
                 background: white;
                 max-width: 100%;
             }
             #maskCanvas {
                 cursor: crosshair;
                 background: transparent;
                 border: none;
             }
            .upload-area {
                border: 3px dashed rgba(255, 255, 255, 0.5);
                border-radius: 10px;
                padding: 30px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                margin-bottom: 20px;
            }
            .upload-area:hover {
                border-color: rgba(255, 255, 255, 0.8);
                background: rgba(255, 255, 255, 0.05);
            }
            .upload-area.dragover {
                border-color: #ffc107;
                background: rgba(255, 193, 7, 0.1);
            }
            .brush-size {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .brush-size input[type="range"] {
                flex: 1;
                padding: 0;
            }
            .brush-size span {
                min-width: 50px;
                font-weight: bold;
                background: rgba(255, 255, 255, 0.2);
                padding: 8px 12px;
                border-radius: 5px;
                text-align: center;
            }
            
            /* Results */
            .result {
                margin-top: 30px;
                text-align: center;
            }
            .result img {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                margin: 20px 0;
            }
            
            /* Status Messages */
            .status {
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                text-align: center;
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
            
            .api-info {
                background: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                font-size: 14px;
            }
            
            @media (max-width: 768px) {
                .workspace {
                    grid-template-columns: 1fr;
                }
                .grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 AI Image Studio</h1>
            <p class="subtitle">SDXL Text-to-Image + SD 2.0 Inpainting</p>
            
            <!-- Tab Navigation -->
            <div class="tabs">
                <div class="tab active" onclick="switchTab('generate')">
                    🖼️ Text-to-Image
                </div>
                <div class="tab" onclick="switchTab('inpaint')">
                    🎯 Inpainting
                </div>
            </div>
            
            <!-- Text-to-Image Tab -->
            <div id="generateTab" class="tab-content active">
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
                    
                    <button type="button" class="toggle-advanced" onclick="toggleAdvanced('generateAdvanced')">
                        ⚙️ Advanced Settings
                    </button>
                    
                    <div id="generateAdvanced" class="advanced">
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
            </div>
            
            <!-- Inpainting Tab -->
            <div id="inpaintTab" class="tab-content">
                <!-- Upload Area -->
                <div class="upload-area" id="uploadArea">
                    <p>📁 Click or drag an image here to start inpainting</p>
                    <input type="file" id="imageInput" accept="image/*" style="display: none;">
                </div>
                
                <!-- Workspace -->
                <div class="workspace" id="workspace" style="display: none;">
                    <!-- Single Canvas with Image + Mask Overlay -->
                    <div class="canvas-container" style="grid-column: span 2;">
                        <h3>🎨 Draw on the image to mark areas you want to regenerate</h3>
                        <div style="position: relative; display: inline-block;">
                            <canvas id="imageCanvas"></canvas>
                            <canvas id="maskCanvas" style="position: absolute; top: 0; left: 0; pointer-events: auto;"></canvas>
                        </div>
                    </div>
                </div>
                
                <!-- Inpainting Controls -->
                <div id="inpaintControls" style="display: none;">
                    <div class="form-group">
                        <label>🖌️ Drawing Tools:</label>
                        <div class="button-group">
                            <button type="button" id="brushTool" class="active">🖌️ Brush</button>
                            <button type="button" id="eraserTool">🧹 Eraser</button>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>📏 Brush Size:</label>
                        <div class="brush-size">
                            <input type="range" id="brushSize" min="5" max="50" value="20">
                            <span id="brushSizeValue">20px</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="inpaintPrompt">✨ Inpainting Prompt:</label>
                        <textarea id="inpaintPrompt" rows="3" placeholder="Be very specific! e.g., 'a red baseball cap with white logo sitting on the dog's head' or 'a blue collar with silver buckle around the dog's neck'"></textarea>
                    </div>
                    
                    <button type="button" class="toggle-advanced" onclick="toggleAdvanced('inpaintAdvanced')">
                        ⚙️ Advanced Settings
                    </button>
                    
                    <div id="inpaintAdvanced" class="advanced">
                        <div class="grid">
                            <div class="form-group">
                                <label for="inpaintSteps">Inference Steps:</label>
                                <select id="inpaintSteps">
                                    <option value="20">20 (Fast)</option>
                                    <option value="30">30 (Balanced)</option>
                                    <option value="50" selected>50 (High Quality)</option>
                                    <option value="75">75 (Max Quality)</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="inpaintGuidance">Guidance Scale:</label>
                                <select id="inpaintGuidance">
                                    <option value="7.5" selected>7.5 (Balanced)</option>
                                    <option value="10.0">10.0 (Strict)</option>
                                    <option value="12.0">12.0 (Very Strict)</option>
                                    <option value="15.0">15.0 (Maximum)</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="inpaintStrength">Strength (how much to change):</label>
                                <select id="inpaintStrength">
                                    <option value="0.7">0.7 (Subtle)</option>
                                    <option value="0.8">0.8 (Moderate)</option>
                                    <option value="0.9">0.9 (Strong)</option>
                                    <option value="0.95" selected>0.95 (Very Strong)</option>
                                    <option value="1.0">1.0 (Maximum)</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <div class="button-group">
                        <button type="button" id="clearMask" class="secondary">🗑️ Clear Mask</button>
                        <button type="button" id="newImage" class="secondary">📁 New Image</button>
                    </div>
                    
                    <div class="button-group" id="debugButtons" style="display: none;">
                        <button type="button" id="downloadImage" class="secondary">📥 Download Image</button>
                        <button type="button" id="downloadMask" class="secondary">📥 Download Mask</button>
                    </div>
                    
                    <div id="maskPreview" style="display: none; margin: 20px 0;">
                        <h4 style="color: white; text-align: center;">🔍 Mask Preview (Black/White)</h4>
                        <p style="color: white; text-align: center; font-size: 14px; opacity: 0.8; margin: 10px 0;">
                            White areas = what will be regenerated | Black areas = what stays the same
                        </p>
                        <div style="text-align: center;">
                            <canvas id="previewCanvas" style="border: 2px solid #ddd; border-radius: 5px; max-width: 200px; background: white;"></canvas>
                        </div>
                        <p style="color: white; text-align: center; font-size: 12px; opacity: 0.7; margin-top: 10px;">
                            Use download buttons below to inspect what's sent to the API
                        </p>
                    </div>
                    
                    <button type="button" id="inpaintBtn">
                        🎯 Start Inpainting
                    </button>
                </div>
            </div>
            
            <!-- Results -->
            <div class="result" id="result"></div>
            
            <!-- API Info -->
            <div class="api-info">
                <strong>📊 API Endpoints:</strong><br>
                • <a href="/health" style="color: #ffc107;">/health</a> - Check server status<br>
                • <a href="/docs" style="color: #ffc107;">/docs</a> - API documentation<br>
                • POST /generate - Generate images<br>
                • POST /inpaint - Inpaint images
            </div>
        </div>

        <script>
            // Global variables
            let currentTab = 'generate';
            let isProcessing = false;
            
                         // Inpainting variables
             let imageCanvas, maskCanvas, imageCtx, maskCtx;
             let isDrawing = false;
             let currentTool = 'brush';
             let brushSize = 20;
             let uploadedImageFile = null;
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                setupEventListeners();
                setupCanvas();
                checkServerHealth();
            });
            
            // Tab Management
            function switchTab(tabName) {
                currentTab = tabName;
                
                // Update tab buttons
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                event.target.classList.add('active');
                
                // Update tab content
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                document.getElementById(tabName + 'Tab').classList.add('active');
            }
            
            function toggleAdvanced(elementId) {
                const advanced = document.getElementById(elementId);
                advanced.classList.toggle('show');
            }
            
            // Text-to-Image Generation
            function setupEventListeners() {
                // Text-to-image form
                document.getElementById('generateForm').addEventListener('submit', handleGenerate);
                
                // Inpainting upload
                const uploadArea = document.getElementById('uploadArea');
                const imageInput = document.getElementById('imageInput');
                
                uploadArea.addEventListener('click', () => imageInput.click());
                uploadArea.addEventListener('dragover', handleDragOver);
                uploadArea.addEventListener('drop', handleDrop);
                imageInput.addEventListener('change', handleImageSelect);
                
                // Inpainting tools
                document.getElementById('brushTool').addEventListener('click', () => setTool('brush'));
                document.getElementById('eraserTool').addEventListener('click', () => setTool('eraser'));
                document.getElementById('brushSize').addEventListener('input', updateBrushSize);
                document.getElementById('clearMask').addEventListener('click', clearMask);
                document.getElementById('newImage').addEventListener('click', resetInpainting);
                document.getElementById('inpaintBtn').addEventListener('click', handleInpaint);
                
                // Debug buttons
                document.getElementById('downloadImage').addEventListener('click', downloadImage);
                document.getElementById('downloadMask').addEventListener('click', downloadMask);
            }
            
            async function handleGenerate(e) {
                e.preventDefault();
                
                if (isProcessing) return;
                
                const prompt = document.getElementById('prompt').value.trim();
                if (!prompt) {
                    showStatus('Please enter a prompt', 'error');
                    return;
                }
                
                isProcessing = true;
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
                    
                    showResult(data.image, data.prompt, data.generation_time, totalTime);
                    showStatus(`✅ Image generated successfully in ${data.generation_time}s`, 'success');
                    
                } catch (error) {
                    console.error('Error:', error);
                    showStatus(`❌ Error: ${error.message}`, 'error');
                } finally {
                    isProcessing = false;
                    generateBtn.disabled = false;
                    generateBtn.textContent = '🚀 Generate Image';
                }
            }
            
                         // Inpainting Functions
             function setupCanvas() {
                 imageCanvas = document.getElementById('imageCanvas');
                 maskCanvas = document.getElementById('maskCanvas');
                 imageCtx = imageCanvas.getContext('2d');
                 maskCtx = maskCanvas.getContext('2d');
                 
                 // Mask canvas drawing events (overlay canvas)
                 maskCanvas.addEventListener('mousedown', startDrawing);
                 maskCanvas.addEventListener('mousemove', draw);
                 maskCanvas.addEventListener('mouseup', stopDrawing);
                 maskCanvas.addEventListener('mouseout', stopDrawing);
                 
                 // Touch events for mobile
                 maskCanvas.addEventListener('touchstart', handleTouch);
                 maskCanvas.addEventListener('touchmove', handleTouch);
                 maskCanvas.addEventListener('touchend', stopDrawing);
             }
            
            function handleDragOver(e) {
                e.preventDefault();
                e.currentTarget.classList.add('dragover');
            }
            
            function handleDrop(e) {
                e.preventDefault();
                e.currentTarget.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    loadImage(files[0]);
                }
            }
            
            function handleImageSelect(e) {
                const files = e.target.files;
                if (files.length > 0) {
                    loadImage(files[0]);
                }
            }
            
            function loadImage(file) {
                if (!file.type.startsWith('image/')) {
                    showStatus('Please select a valid image file', 'error');
                    return;
                }
                
                uploadedImageFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = new Image();
                    img.onload = function() {
                        setupImageCanvas(img);
                    };
                    img.src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
            
                         function setupImageCanvas(img) {
                 // Calculate canvas size (max 512px while maintaining aspect ratio for SD 2.0)
                 const maxSize = 512;
                 let canvasWidth = img.width;
                 let canvasHeight = img.height;
                 
                 if (img.width > maxSize || img.height > maxSize) {
                     const ratio = Math.min(maxSize / img.width, maxSize / img.height);
                     canvasWidth = img.width * ratio;
                     canvasHeight = img.height * ratio;
                 }
                 
                 // Set both canvas dimensions to match
                 imageCanvas.width = canvasWidth;
                 imageCanvas.height = canvasHeight;
                 maskCanvas.width = canvasWidth;
                 maskCanvas.height = canvasHeight;
                 
                 // Draw original image on background canvas
                 imageCtx.drawImage(img, 0, 0, canvasWidth, canvasHeight);
                 
                 // Clear mask canvas and make it transparent
                 maskCtx.clearRect(0, 0, canvasWidth, canvasHeight);
                 
                                 // Show workspace and tools
                document.getElementById('uploadArea').style.display = 'none';
                document.getElementById('workspace').style.display = 'grid';
                document.getElementById('inpaintControls').style.display = 'block';
                document.getElementById('debugButtons').style.display = 'grid';
                document.getElementById('maskPreview').style.display = 'block';
                
                // Initialize mask preview
                updateMaskPreview();
                
                showStatus('✅ Image loaded! Draw red areas to mark what you want to regenerate', 'success');
             }
            
            function setTool(tool) {
                currentTool = tool;
                document.getElementById('brushTool').classList.toggle('active', tool === 'brush');
                document.getElementById('eraserTool').classList.toggle('active', tool === 'eraser');
                
                maskCanvas.style.cursor = tool === 'brush' ? 'crosshair' : 'grab';
            }
            
            function updateBrushSize() {
                brushSize = document.getElementById('brushSize').value;
                document.getElementById('brushSizeValue').textContent = brushSize + 'px';
            }
            
            function getMousePos(e) {
                const rect = maskCanvas.getBoundingClientRect();
                return {
                    x: (e.clientX - rect.left) * (maskCanvas.width / rect.width),
                    y: (e.clientY - rect.top) * (maskCanvas.height / rect.height)
                };
            }
            
            function startDrawing(e) {
                isDrawing = true;
                const pos = getMousePos(e);
                drawOnMask(pos.x, pos.y);
            }
            
            function draw(e) {
                if (!isDrawing) return;
                const pos = getMousePos(e);
                drawOnMask(pos.x, pos.y);
            }
            
            function stopDrawing() {
                isDrawing = false;
            }
            
            function handleTouch(e) {
                e.preventDefault();
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
                    clientX: touch.clientX,
                    clientY: touch.clientY
                });
                maskCanvas.dispatchEvent(mouseEvent);
            }
            
                                     function drawOnMask(x, y) {
                if (currentTool === 'brush') {
                    maskCtx.globalCompositeOperation = 'source-over';
                    maskCtx.fillStyle = 'rgba(255, 0, 0, 0.5)'; // Semi-transparent red
                } else {
                    maskCtx.globalCompositeOperation = 'destination-out'; // Eraser
                }
                
                maskCtx.beginPath();
                maskCtx.arc(x, y, brushSize / 2, 0, 2 * Math.PI);
                maskCtx.fill();
                
                // Update mask preview
                updateMaskPreview();
            }
            
                                     function clearMask() {
                maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                updateMaskPreview();
                showStatus('🗑️ Mask cleared', 'success');
            }
            
            function resetInpainting() {
                document.getElementById('uploadArea').style.display = 'block';
                document.getElementById('workspace').style.display = 'none';
                document.getElementById('inpaintControls').style.display = 'none';
                document.getElementById('debugButtons').style.display = 'none';
                document.getElementById('maskPreview').style.display = 'none';
                document.getElementById('result').innerHTML = '';
                document.getElementById('imageInput').value = '';
                uploadedImageFile = null;
            }
            
            async function handleInpaint() {
                const prompt = document.getElementById('inpaintPrompt').value.trim();
                if (!prompt) {
                    showStatus('Please enter a prompt describing what you want to generate', 'error');
                    return;
                }
                
                if (!uploadedImageFile) {
                    showStatus('Please upload an image first', 'error');
                    return;
                }
                
                                 // Check if mask has any red areas
                 const imageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
                 const hasRedPixels = Array.from(imageData.data).some((pixel, index) => 
                     index % 4 === 3 && pixel > 0 // Check alpha channel for drawn areas
                 );
                 
                 if (!hasRedPixels) {
                     showStatus('Please paint some red areas on the mask to indicate what to regenerate', 'error');
                     return;
                 }
                
                isProcessing = true;
                const inpaintBtn = document.getElementById('inpaintBtn');
                inpaintBtn.disabled = true;
                inpaintBtn.textContent = '⏳ Inpainting...';
                
                showStatus('🎨 Starting inpainting... This may take 30-60 seconds', 'loading');
                
                                 try {
                     const startTime = Date.now();
                     
                     // Convert red overlay to black/white mask
                     const bwMaskCanvas = createBlackWhiteMask();
                     const maskBlob = await canvasToBlob(bwMaskCanvas);
                    
                    // Create FormData
                    const formData = new FormData();
                    formData.append('image', uploadedImageFile);
                    formData.append('mask', maskBlob, 'mask.png');
                    formData.append('prompt', prompt);
                    formData.append('num_inference_steps', document.getElementById('inpaintSteps').value);
                    formData.append('guidance_scale', document.getElementById('inpaintGuidance').value);
                    formData.append('strength', document.getElementById('inpaintStrength').value);
                    
                    const response = await fetch('/inpaint', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Inpainting failed');
                    }
                    
                    const data = await response.json();
                    const totalTime = Date.now() - startTime;
                    
                    showResult(data.image, data.prompt, data.generation_time, totalTime);
                    showStatus('✅ Inpainting completed successfully!', 'success');
                    
                } catch (error) {
                    console.error('Error:', error);
                    showStatus(`❌ Error: ${error.message}`, 'error');
                } finally {
                    isProcessing = false;
                    inpaintBtn.disabled = false;
                    inpaintBtn.textContent = '🎯 Start Inpainting';
                }
            }
            
                         function createBlackWhiteMask() {
                 // Create a temporary canvas for black/white mask
                 const tempCanvas = document.createElement('canvas');
                 tempCanvas.width = maskCanvas.width;
                 tempCanvas.height = maskCanvas.height;
                 const tempCtx = tempCanvas.getContext('2d');
                 
                 // Fill with black background
                 tempCtx.fillStyle = 'black';
                 tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
                 
                 // Get the red overlay data
                 const imageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
                 const data = imageData.data;
                 
                 // Create white areas where red was drawn
                 for (let i = 0; i < data.length; i += 4) {
                     if (data[i + 3] > 0) { // If alpha > 0 (red was drawn)
                         const x = (i / 4) % maskCanvas.width;
                         const y = Math.floor(i / 4 / maskCanvas.width);
                         
                         tempCtx.fillStyle = 'white';
                         tempCtx.fillRect(x, y, 1, 1);
                     }
                 }
                 
                 return tempCanvas;
             }
             
                         function canvasToBlob(canvas) {
                return new Promise(resolve => {
                    canvas.toBlob(resolve, 'image/png');
                });
            }
            
            // Debug Functions
            function updateMaskPreview() {
                if (!maskCanvas) return;
                
                const previewCanvas = document.getElementById('previewCanvas');
                const previewCtx = previewCanvas.getContext('2d');
                
                // Set preview canvas size (small version)
                const scale = 200 / Math.max(maskCanvas.width, maskCanvas.height);
                previewCanvas.width = maskCanvas.width * scale;
                previewCanvas.height = maskCanvas.height * scale;
                
                // Create and draw the black/white mask
                const bwMask = createBlackWhiteMask();
                previewCtx.drawImage(bwMask, 0, 0, previewCanvas.width, previewCanvas.height);
            }
            
            function downloadImage() {
                if (!imageCanvas) {
                    showStatus('No image to download', 'error');
                    return;
                }
                
                imageCanvas.toBlob(blob => {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `processed_image_${Date.now()}.png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    showStatus('📥 Image downloaded', 'success');
                }, 'image/png');
            }
            
            function downloadMask() {
                if (!maskCanvas) {
                    showStatus('No mask to download', 'error');
                    return;
                }
                
                const bwMask = createBlackWhiteMask();
                bwMask.toBlob(blob => {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `mask_${Date.now()}.png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    showStatus('📥 Mask downloaded', 'success');
                }, 'image/png');
            }
            
            // Utility Functions
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
            
            function showResult(base64Image, prompt, generationTime, totalTime) {
                const result = document.getElementById('result');
                
                // Remove previous image
                const existingImage = result.querySelector('img');
                if (existingImage) {
                    existingImage.remove();
                }
                
                const img = document.createElement('img');
                img.src = `data:image/png;base64,${base64Image}`;
                img.alt = 'Generated Image';
                
                const info = document.createElement('div');
                info.innerHTML = `
                    <strong>📝 Prompt:</strong> ${prompt}<br>
                    <strong>⏱️ Generation Time:</strong> ${generationTime}s<br>
                    <strong>🌐 Total Request Time:</strong> ${(totalTime/1000).toFixed(2)}s
                `;
                info.style.marginTop = '15px';
                info.style.background = 'rgba(255, 255, 255, 0.1)';
                info.style.padding = '15px';
                info.style.borderRadius = '8px';
                
                result.appendChild(img);
                result.appendChild(info);
            }
            
            async function checkServerHealth() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    
                                         if (data.status === 'healthy' && data.text_to_image_loaded && data.inpainting_loaded) {
                         showStatus('🟢 Server is healthy - SDXL + SD 2.0 Inpainting loaded', 'success');
                     } else if (data.status === 'healthy' && data.text_to_image_loaded) {
                         showStatus('🟡 SDXL ready, SD 2.0 Inpainting loading...', 'loading');
                    } else {
                        showStatus('🟡 Server responding but models may not be loaded', 'error');
                    }
                } catch (error) {
                    showStatus('🔴 Cannot connect to server', 'error');
                }
            }
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
            "inpaint": "POST /inpaint - Inpaint masked areas of an image",
            "health": "GET /health - Health check with GPU info",
            "docs": "GET /docs - API documentation",
            "web": "GET / - Web UI interface"
        },
                "models": {
            "text_to_image": "stabilityai/stable-diffusion-xl-base-1.0",
            "inpainting": "stabilityai/stable-diffusion-2-inpainting"
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