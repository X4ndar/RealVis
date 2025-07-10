#!/usr/bin/env python3
"""
RealVis 5 FastAPI Backend - Local Model Execution
FastAPI backend that runs RealVis 5 models locally for frontend communication.
"""

import logging
import base64
import io
import torch
import time
import os
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting
from PIL import Image
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Security
security = HTTPBearer(auto_error=False)

# Configuration from environment variables
API_KEY = os.getenv("API_KEY")  # Optional - leave empty to disable auth
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# CORS Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
ALLOWED_METHODS = os.getenv("ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
ALLOWED_HEADERS = os.getenv("ALLOWED_HEADERS", "*").split(",")
ALLOW_CREDENTIALS = os.getenv("ALLOW_CREDENTIALS", "true").lower() == "true"

# Model Configuration
REALVIS_MODEL = os.getenv("REALVIS_MODEL", "GraydientPlatformAPI/realvis5light-xl")
INPAINTING_MODEL = os.getenv("INPAINTING_MODEL", "stabilityai/stable-diffusion-2-inpainting")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")

# Generation Defaults & Limits
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "25"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "2.0"))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_NEGATIVE_PROMPT = os.getenv("DEFAULT_NEGATIVE_PROMPT", "cartoon, anime, painted, artificial, low quality, blurry, distorted")

MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))
MAX_WIDTH = int(os.getenv("MAX_WIDTH", "1536"))
MAX_HEIGHT = int(os.getenv("MAX_HEIGHT", "1536"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "10"))

# GPU Optimizations
ENABLE_GPU_OPTIMIZATIONS = os.getenv("ENABLE_GPU_OPTIMIZATIONS", "true").lower() == "true"
ENABLE_XFORMERS = os.getenv("ENABLE_XFORMERS", "true").lower() == "true"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "true").lower() == "true"
ENABLE_ATTENTION_SLICING = os.getenv("ENABLE_ATTENTION_SLICING", "true").lower() == "true"
ENABLE_VAE_SLICING = os.getenv("ENABLE_VAE_SLICING", "true").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_GPU_MEMORY = os.getenv("LOG_GPU_MEMORY", "true").lower() == "true"

# Deployment URLs
API_BASE_URL = os.getenv("API_BASE_URL", "https://do9n3s330iext0-8000.proxy.runpod.net")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Global variables to store the pipelines
pipeline = None
inpaint_pipeline = None

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt to avoid unwanted elements")
    num_inference_steps: int = Field(default=DEFAULT_STEPS, ge=1, le=MAX_STEPS, description="Number of inference steps")
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, ge=0.1, le=20.0, description="Guidance scale for prompt adherence")
    height: int = Field(default=DEFAULT_HEIGHT, ge=64, le=MAX_HEIGHT, description="Image height in pixels")
    width: int = Field(default=DEFAULT_WIDTH, ge=64, le=MAX_WIDTH, description="Image width in pixels")

class GenerateResponse(BaseModel):
    image: str = Field(..., description="Base64 encoded generated image")
    prompt: str = Field(..., description="Text prompt used for generation")
    negative_prompt: Optional[str] = Field(..., description="Negative prompt used")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    model_used: str = Field(..., description="Model used for generation")

class InpaintRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for inpainting")
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt to avoid unwanted elements")
    num_inference_steps: int = Field(default=30, ge=1, le=MAX_STEPS, description="Number of inference steps")
    guidance_scale: float = Field(default=8.0, ge=0.1, le=20.0, description="Guidance scale for prompt adherence")
    strength: float = Field(default=0.95, ge=0.1, le=1.0, description="Inpainting strength")

class InpaintResponse(BaseModel):
    image: str = Field(..., description="Base64 encoded inpainted image")
    prompt: str = Field(..., description="Text prompt used for inpainting")
    negative_prompt: Optional[str] = Field(..., description="Negative prompt used")
    generation_time: float = Field(..., description="Time taken for inpainting in seconds")
    model_used: str = Field(..., description="Model used for inpainting")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Server health status")
    realvis_loaded: bool = Field(..., description="Whether RealVis 5 model is loaded")
    inpainting_loaded: bool = Field(..., description="Whether inpainting model is loaded")
    cuda_available: bool = Field(..., description="Whether CUDA is available")
    gpu_info: dict = Field(..., description="GPU information")
    api_base_url: str = Field(..., description="API base URL")
    api_version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")

# Authentication function
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key if authentication is enabled."""
    if API_KEY:  # Only check if API_KEY is set in environment
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="API key required. Include 'Authorization: Bearer YOUR_API_KEY' in headers."
            )
        if credentials.credentials != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key."
            )
    return True

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
    if not LOG_GPU_MEMORY or not torch.cuda.is_available():
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

def load_realvis_model():
    """Load the RealVis 5 model with optimized settings."""
    logger.info(f"Loading RealVis 5 model: {REALVIS_MODEL}")
    
    try:
        cache_dir = MODEL_CACHE_DIR if MODEL_CACHE_DIR else None
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            REALVIS_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=cache_dir
        )
        
        logger.info("Moving model to GPU...")
        pipe = pipe.to("cuda")
        
        # Apply GPU optimizations if enabled
        if ENABLE_GPU_OPTIMIZATIONS:
            # Enable xFormers if available and enabled
            if ENABLE_XFORMERS and hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xFormers memory efficient attention")
                except:
                    logger.info("xFormers not available, using default attention")
            
            # Enable model CPU offload
            if ENABLE_CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload for memory optimization")
            
            # Enable attention slicing
            if ENABLE_ATTENTION_SLICING:
                try:
                    pipe.enable_attention_slicing()
                    logger.info("Enabled attention slicing")
                except:
                    logger.info("Attention slicing not available")
                
            # Enable VAE slicing
            if ENABLE_VAE_SLICING:
                try:
                    pipe.enable_vae_slicing()
                    logger.info("Enabled VAE slicing")
                except:
                    logger.info("VAE slicing not available")
        
        logger.info("RealVis 5 model loaded successfully with optimizations!")
        print_gpu_usage("(After RealVis Model Loading)")
        
        return pipe
        
    except Exception as e:
        logger.error(f"Error loading RealVis 5 model: {e}")
        return None

def load_inpainting_model():
    """Load the inpainting model with optimized settings."""
    logger.info(f"Loading inpainting model: {INPAINTING_MODEL}")
    
    try:
        cache_dir = MODEL_CACHE_DIR if MODEL_CACHE_DIR else None
        
        pipe = AutoPipelineForInpainting.from_pretrained(
            INPAINTING_MODEL,
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        ).to("cuda")
        
        # Apply same optimizations as RealVis model
        if ENABLE_GPU_OPTIMIZATIONS:
            if ENABLE_XFORMERS and hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xFormers for inpainting")
                except:
                    logger.info("xFormers not available for inpainting")
            
            if ENABLE_CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload for inpainting")
            
            if ENABLE_ATTENTION_SLICING:
                try:
                    pipe.enable_attention_slicing()
                    logger.info("Enabled attention slicing for inpainting")
                except:
                    logger.info("Attention slicing not available for inpainting")
                
            if ENABLE_VAE_SLICING:
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
    title="RealVis 5 API",
    description="Advanced photorealistic image generation API using RealVis 5 model. Local backend for frontend communication.",
    version="4.0.0",
    docs_url="/docs" if DEBUG else "/docs",
    redoc_url="/redoc" if DEBUG else "/redoc"
)

# Configure CORS
if ALLOWED_ORIGINS == ["*"]:
    origins = ["*"]
else:
    origins = ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS if ALLOWED_HEADERS != ["*"] else ["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load both RealVis 5 and inpainting pipelines at startup."""
    global pipeline, inpaint_pipeline
    
    logger.info("Starting RealVis 5 API...")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"API Base URL: {API_BASE_URL}")
    logger.info(f"Authentication: {'Enabled' if API_KEY else 'Disabled'}")
    logger.info(f"Allowed CORS Origins: {ALLOWED_ORIGINS}")
    
    # Check GPU availability
    if not check_gpu_availability():
        raise RuntimeError("CUDA is not available. This app requires a GPU.")
    
    # Initial GPU usage baseline
    print_gpu_usage("(Initial Baseline)")
    
    # Load RealVis 5 model
    pipeline = load_realvis_model()
    if pipeline is None:
        raise RuntimeError("Failed to load RealVis 5 pipeline")
    
    # Load inpainting model
    inpaint_pipeline = load_inpainting_model()
    if inpaint_pipeline is None:
        logger.warning("Failed to load inpainting pipeline. Inpainting features will be disabled.")
    
    logger.info("RealVis 5 API startup completed successfully!")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with GPU info and configuration details."""
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
    
    return HealthResponse(
        status="healthy",
        realvis_loaded=pipeline is not None,
        inpainting_loaded=inpaint_pipeline is not None,
        cuda_available=torch.cuda.is_available(),
        gpu_info=gpu_info,
        api_base_url=API_BASE_URL,
        api_version="4.0.0",
        environment=ENVIRONMENT
    )

@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
async def generate_image(request: GenerateRequest) -> GenerateResponse:
    """Generate a photorealistic image using RealVis 5 model."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="RealVis 5 model not loaded. Please check server logs."
        )
    
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty."
        )
    
    # Use provided negative prompt or default
    negative_prompt = request.negative_prompt or DEFAULT_NEGATIVE_PROMPT
    
    logger.info(f"Generating image from prompt: '{prompt}'")
    logger.info(f"Parameters: steps={request.num_inference_steps}, guidance={request.guidance_scale}, size={request.width}x{request.height}")
    
    # Check GPU usage before generation
    print_gpu_usage("(Before Image Generation)")
    
    try:
        start_time = time.time()
        
        # Generate image with RealVis 5 optimized parameters
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
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
            negative_prompt=negative_prompt,
            generation_time=round(generation_time, 2),
            model_used=REALVIS_MODEL
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

@app.post("/inpaint", response_model=InpaintResponse, dependencies=[Depends(verify_api_key)])
async def inpaint_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(8.0),
    strength: float = Form(0.95)
):
    """Inpaint an image using the inpainting model."""
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
    
    # Use provided negative prompt or default
    neg_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
    
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
                negative_prompt=neg_prompt,
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
            negative_prompt=neg_prompt,
            generation_time=round(generation_time, 2),
            model_used=INPAINTING_MODEL
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

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "RealVis 5 API - Advanced Photorealistic Image Generation",
        "version": "4.0.0",
        "mode": "local_execution",
        "api_base_url": API_BASE_URL,
        "environment": ENVIRONMENT,
        "authentication": "enabled" if API_KEY else "disabled",
        "endpoints": {
            "generate": "POST /generate - Generate photorealistic images",
            "inpaint": "POST /inpaint - Inpaint masked areas of images",
            "health": "GET /health - Health check with GPU info",
            "docs": "GET /docs - Interactive API documentation",
            "redoc": "GET /redoc - Alternative API documentation"
        },
        "models": {
            "realvis": REALVIS_MODEL,
            "inpainting": INPAINTING_MODEL
        },
        "features": [
            "RealVis 5 photorealistic generation",
            "Advanced inpainting capabilities",
            "GPU memory optimizations",
            "API key authentication",
            "CORS support for frontends",
            "Comprehensive error handling"
        ],
        "optimizations": {
            "gpu_optimizations": ENABLE_GPU_OPTIMIZATIONS,
            "xformers": ENABLE_XFORMERS,
            "cpu_offload": ENABLE_CPU_OFFLOAD,
            "attention_slicing": ENABLE_ATTENTION_SLICING,
            "vae_slicing": ENABLE_VAE_SLICING
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RealVis 5 API - Advanced Photorealistic Image Generation",
        "version": "4.0.0",
        "api_base_url": API_BASE_URL,
        "environment": ENVIRONMENT,
        "mode": "local_execution",
        "documentation": "/docs",
        "api_info": "/api",
        "health_check": "/health"
    }

if __name__ == "__main__":
    # Run the app with uvicorn
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL.lower(),
        reload=DEBUG
    ) 