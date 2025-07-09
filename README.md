# 🎨 AI Image Studio

A complete FastAPI-powered image generation and inpainting system using Stable Diffusion models. Features both text-to-image generation and advanced inpainting with an intuitive web interface.

## Features

### 🖼️ Text-to-Image Generation
- 🚀 **SDXL Model**: `stabilityai/stable-diffusion-xl-base-1.0`
- 🎯 **High Quality**: 1024x1024 native resolution
- ⚡ **GPU Optimized**: FP16 precision with memory optimizations

### 🎨 Advanced Inpainting
- 🔧 **SD 2.0 Inpainting**: `stabilityai/stable-diffusion-2-inpainting`
- 🖌️ **Interactive Canvas**: Draw directly on images to mark areas for regeneration
- 🎯 **Precise Control**: Brush and eraser tools with adjustable sizes
- 📱 **Touch Support**: Works on mobile devices

### 🌐 Web Interface
- 💻 **Beautiful UI**: Modern glass-morphism design
- 📊 **Real-time Status**: Generation progress and GPU monitoring
- 🔄 **Tabbed Interface**: Switch between text-to-image and inpainting
- 📥 **Debug Tools**: Download processed images and masks

### ⚡ Performance Optimizations
- 🧠 **Memory Efficient**: xformers attention, CPU offload, attention slicing
- 🔄 **GPU Cache Management**: Automatic cleanup after generation
- 📊 **Resource Monitoring**: Real-time GPU usage tracking
- 🚀 **FastAPI Backend**: High-performance async API

## Requirements

- **GPU**: CUDA-compatible GPU with 12GB+ VRAM (16GB recommended)
- **Python**: 3.8+
- **Dependencies**: See `requirements.txt`

## Quick Start

### 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd InprintSelfHost
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 🌐 Start the Web Interface

1. **Launch the FastAPI server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   ```
   http://localhost:8000
   ```
   or if accessing remotely:
   ```
   http://YOUR_SERVER_IP:8000
   ```

3. **Start creating!** 🎨
   - **Text-to-Image**: Enter prompts and generate images
   - **Inpainting**: Upload images, draw masks, and regenerate specific areas

### 📡 Alternative: Command Line (Legacy)

For the original command-line interface:
```bash
python generate_image.py
```

## Example Usage

### 🌐 Web Interface

```bash
$ python app.py
Loading Stable Diffusion XL model...
Model: stabilityai/stable-diffusion-xl-base-1.0
✅ CUDA is available
📊 GPU Device: NVIDIA RTX 4090
💾 GPU Memory: 24.0 GB
⚡ Enabled xformers memory efficient attention
✅ Model loaded successfully!

Loading Stable Diffusion 2.0 Inpainting model...
Model: stabilityai/stable-diffusion-2-inpainting
✅ Inpainting model loaded successfully!

INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Then open your browser to `http://localhost:8000` and enjoy the web interface!**

### 📡 API Usage

#### Text-to-Image Generation
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a majestic dragon flying over a cyberpunk city at sunset",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024
  }'
```

#### Inpainting
```bash
curl -X POST "http://localhost:8000/inpaint" \
  -F "image=@original.jpg" \
  -F "mask=@mask.png" \
  -F "prompt=a red baseball cap on the dog's head" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=7.5" \
  -F "strength=0.95"
```

#### Health Check
```bash
curl http://localhost:8000/health
```

## Configuration

### 🌐 Web Interface Settings

All parameters can be adjusted through the web interface:

#### Text-to-Image Parameters
- **Inference Steps**: 15-50 (default: 20)
- **Guidance Scale**: 5.0-15.0 (default: 7.5)
- **Dimensions**: 512px-1024px (default: 1024x1024)

#### Inpainting Parameters  
- **Inference Steps**: 20-75 (default: 50)
- **Guidance Scale**: 7.5-15.0 (default: 7.5)
- **Strength**: 0.7-1.0 (default: 0.95) - How much to change masked areas
- **Brush Size**: 5-50px (default: 20px)

### 🔧 Server Configuration

Edit `app.py` to modify server settings:

```python
# Change server host/port
uvicorn.run(
    "app:app",
    host="0.0.0.0",    # Change to "127.0.0.1" for local only
    port=8000,         # Change port if needed
    log_level="info",
    reload=False       # Set to True for development
)
```

### 🧠 Memory Optimization

GPU optimizations are automatically enabled:
- ✅ xformers memory efficient attention
- ✅ Model CPU offload
- ✅ Attention slicing
- ✅ VAE slicing
- ✅ Automatic GPU cache clearing

For lower VRAM GPUs, you can disable CPU offload in the model loading functions.

## 📡 API Endpoints

- **GET** `/` - Web UI interface
- **GET** `/health` - Server and GPU status
- **GET** `/docs` - Automatic API documentation (Swagger UI)
- **POST** `/generate` - Text-to-image generation
- **POST** `/inpaint` - Image inpainting

## 🎯 Web Interface Features

### Text-to-Image Tab
- Enter prompts and generate high-quality 1024x1024 images
- Advanced settings for fine-tuning generation
- Real-time generation status and timing

### Inpainting Tab  
- Upload images with drag & drop support
- Interactive canvas with brush and eraser tools
- Real-time mask preview (black/white)
- Download processed images and masks for debugging
- Touch support for mobile devices

## Troubleshooting

### 🚨 Common Issues

**Server won't start:**
```bash
# Install missing dependency
pip install python-multipart
```

**CUDA not available:**
- Ensure you're running in a GPU-enabled environment
- Check CUDA installation: `nvidia-smi`

**Out of memory:**
- Close other GPU applications
- Reduce image resolution in advanced settings
- Try text-to-image with smaller dimensions first

**Inpainting not working:**
- Check mask preview - white areas should be clearly visible
- Use more specific prompts (e.g., "red baseball cap with white logo")
- Try higher strength values (0.9-1.0)
- Download mask to verify it's correct

**Models downloading slowly:**
- First run downloads ~8GB total (SDXL + SD 2.0 models)
- Subsequent runs use cached models
- Ensure stable internet connection

### 🔗 Remote Access

**SSH Port Forwarding (Secure):**
```bash
ssh -L 8000:localhost:8000 user@your-server
```

**Direct IP Access:**
- Server runs on `http://0.0.0.0:8000` by default
- Access via `http://YOUR_SERVER_IP:8000`
- **Security Note**: Add authentication for production use

## Notes

- 🕐 **Generation Times**: 20-60s for text-to-image, 30-90s for inpainting
- 💾 **Model Cache**: ~/.cache/huggingface/transformers/
- 🔄 **GPU Memory**: Automatically cleared after each generation
- 📱 **Mobile Support**: Full touch interface for tablets/phones
- 🎨 **Image Formats**: Input (JPG/PNG), Output (PNG with base64 encoding)

## 📁 Project Structure

```
InprintSelfHost/
├── app.py                 # 🚀 Main FastAPI application with web UI
├── generate_image.py      # 📡 Legacy command-line generator
├── requirements.txt       # 📦 Python dependencies
└── README.md             # 📚 This documentation
```

## 🎯 Quick Summary

This is a **complete AI image generation solution** that combines:

1. **🖼️ High-Quality Text-to-Image**: SDXL model for stunning 1024x1024 images
2. **🎨 Professional Inpainting**: Interactive canvas with SD 2.0 for precise edits
3. **🌐 Beautiful Web Interface**: Modern UI with real-time feedback
4. **⚡ GPU Optimized**: Memory efficient with automatic resource management
5. **📡 REST API**: Use programmatically or integrate into other applications

**Perfect for**: Artists, developers, researchers, or anyone wanting professional AI image generation with full control and no API costs!

---

**🌟 Enjoy creating amazing images with AI! 🎨** 