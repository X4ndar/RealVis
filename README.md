# Stable Diffusion XL Text-to-Image Generator

A Python script for generating images from text prompts using Stable Diffusion XL in a GPU-enabled environment.

## Features

- 🚀 Uses `stabilityai/stable-diffusion-xl-base-1.0` model
- ⚡ Optimized for GPU with FP16 precision
- 🎨 Interactive prompt input or hardcoded prompts
- 💾 Automatic image saving with timestamps
- 🧹 Memory management and error handling
- 📊 GPU status and progress monitoring

## Requirements

- CUDA-compatible GPU
- Python 3.8+
- Required packages (see `requirements.txt`)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the script:**
   ```bash
   python generate_image.py
   ```
   or
   ```bash
   ./generate_image.py
   ```

3. **Enter your prompt** when prompted, or modify the script to use hardcoded prompts.

## Example Usage

```bash
$ python generate_image.py
🔥 Starting Stable Diffusion XL Image Generation
============================================================
✅ CUDA is available
📊 GPU Device: NVIDIA A100-SXM4-40GB
💾 GPU Memory: 40.0 GB
🚀 Loading Stable Diffusion XL model...
📦 Model: stabilityai/stable-diffusion-xl-base-1.0
🔄 Moving model to GPU...
⚡ Enabled xformers memory efficient attention
✅ Model loaded successfully!

==================================================
🎨 TEXT-TO-IMAGE GENERATOR
==================================================
📝 Enter your image prompt: A majestic dragon flying over a cyberpunk city at sunset

🎯 Generating image from prompt: 'A majestic dragon flying over a cyberpunk city at sunset'
⏳ This may take 20-60 seconds depending on your GPU...
💾 Image saved as: output_20231201_143022.png
📍 Full path: /workspace/output_20231201_143022.png
💾 Also saved as: output.png

🎉 Image generation completed successfully!
📝 Prompt: A majestic dragon flying over a cyberpunk city at sunset
🖼️  Output: output_20231201_143022.png
🧹 GPU cache cleared
```

## Configuration

### Hardcoded Prompts
To use hardcoded prompts instead of interactive input, uncomment this line in `get_prompt()`:
```python
return "An astronaut riding a green horse"
```

### Memory Optimization
If you run into GPU memory issues, uncomment this line in `load_model()`:
```python
pipe.enable_model_cpu_offload()
```

### Generation Parameters
Modify these parameters in `generate_image()` for different results:
- `num_inference_steps`: Higher = better quality, slower (default: 30)
- `guidance_scale`: Higher = more prompt adherence (default: 7.5)
- `height`/`width`: Image dimensions (default: 1024x1024)

## Output

- Images are saved with timestamps: `output_YYYYMMDD_HHMMSS.png`
- Also saved as `output.png` for easy access
- Full file paths are displayed in the console

## Troubleshooting

- **CUDA not available**: Ensure you're running in a GPU-enabled environment
- **Out of memory**: Try enabling `model_cpu_offload()` or reducing image resolution
- **Model download fails**: Check internet connection and Hugging Face access

## Notes

- First run will download the model (~6GB)
- Subsequent runs will use cached model files
- Generation typically takes 20-60 seconds depending on GPU
- Press Ctrl+C to interrupt generation 