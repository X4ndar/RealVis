#!/usr/bin/env python3
"""
Text-to-Image Generation Script using Stable Diffusion XL
Runs in a GPU-enabled pod for terminal testing.
"""

from diffusers import DiffusionPipeline
import torch
import os
import sys
import threading
import time
from datetime import datetime

def check_gpu_availability():
    """Check if CUDA is available and print GPU info."""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This script requires GPU access.")
        return False
    
    print(f"✅ CUDA is available")
    print(f"📊 GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
    
    print(f"\n📊 GPU Memory Usage {stage}:")
    print(f"   🔹 Total Memory: {total_memory / 1e9:.2f} GB")
    print(f"   🔹 Allocated: {allocated / 1e9:.2f} GB ({allocated/total_memory*100:.1f}%)")
    print(f"   🔹 Reserved: {reserved / 1e9:.2f} GB ({reserved/total_memory*100:.1f}%)")
    print(f"   🔹 Free: {free_memory / 1e9:.2f} GB ({free_memory/total_memory*100:.1f}%)")
    
    # Get additional memory info if available
    try:
        memory_summary = torch.cuda.memory_summary(device)
        cached = torch.cuda.memory_cached(device) if hasattr(torch.cuda, 'memory_cached') else 0
        if cached:
            print(f"   🔹 Cached: {cached / 1e9:.2f} GB")
    except:
        pass

def get_gpu_usage_compact():
    """Get compact GPU memory usage for real-time monitoring."""
    if not torch.cuda.is_available():
        return "No CUDA"
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    
    allocated_gb = allocated / 1e9
    reserved_gb = reserved / 1e9
    total_gb = total_memory / 1e9
    usage_percent = (reserved / total_memory) * 100
    
    return f"📊 GPU: {allocated_gb:.1f}GB/{reserved_gb:.1f}GB/{total_gb:.1f}GB ({usage_percent:.1f}%)"

class GPUMonitor:
    """Real-time GPU memory monitor that runs in a separate thread."""
    
    def __init__(self, interval=2.0):
        self.interval = interval
        self.monitoring = False
        self.thread = None
        self.start_time = None
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            if torch.cuda.is_available():
                elapsed = time.time() - self.start_time if self.start_time else 0
                usage_info = get_gpu_usage_compact()
                print(f"\r⏱️  {elapsed:.0f}s | {usage_info}", end="", flush=True)
            time.sleep(self.interval)
    
    def start(self):
        """Start real-time monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            print("🔄 Starting real-time GPU monitoring...")
    
    def stop(self):
        """Stop real-time monitoring."""
        if self.monitoring:
            self.monitoring = False
            if self.thread:
                self.thread.join(timeout=1.0)
            print("\n🛑 Stopped real-time GPU monitoring")

def load_model():
    """Load the Stable Diffusion XL model with optimized settings."""
    print("🚀 Loading Stable Diffusion XL model...")
    print("📦 Model: stabilityai/stable-diffusion-xl-base-1.0")
    
    # Start monitoring during model loading
    gpu_monitor = GPUMonitor(interval=2.0)
    
    try:
        gpu_monitor.start()
        
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        print("\n🔄 Moving model to GPU...")
        pipe = pipe.to("cuda")
        
        # Enable memory efficient attention for older PyTorch versions
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("⚡ Enabled xformers memory efficient attention")
            except:
                print("⚠️  xformers not available, using default attention")
        
        # Enable memory optimizations for low VRAM GPUs
        pipe.enable_model_cpu_offload()
        print("💾 Enabled model CPU offload for memory optimization")
        
        # Enable memory efficient attention
        try:
            pipe.enable_attention_slicing()
            print("⚡ Enabled attention slicing")
        except:
            print("⚠️  Attention slicing not available")
            
        # Enable VAE slicing for lower memory usage
        try:
            pipe.enable_vae_slicing()
            print("🔧 Enabled VAE slicing")
        except:
            print("⚠️  VAE slicing not available")
        
        # Stop monitoring after model setup
        gpu_monitor.stop()
        
        print("✅ Model loaded successfully with memory optimizations!")
        
        # Print GPU usage after model loading
        print_gpu_usage("(After Model Loading)")
        
        return pipe
        
    except Exception as e:
        gpu_monitor.stop()
        print(f"❌ Error loading model: {e}")
        return None

def get_prompt():
    """Get text prompt from user input."""
    print("\n" + "="*50)
    print("🎨 TEXT-TO-IMAGE GENERATOR")
    print("="*50)
    
    # You can uncomment this for hardcoded testing:
    # return "An astronaut riding a green horse"
    
    prompt = input("📝 Enter your image prompt: ").strip()
    
    if not prompt:
        print("⚠️  No prompt provided, using default...")
        return "An astronaut riding a green horse"
    
    return prompt

def generate_image(pipe, prompt):
    """Generate image from text prompt."""
    print(f"\n🎯 Generating image from prompt: '{prompt}'")
    print("⏳ This may take 20-60 seconds depending on your GPU...")
    
    # Check GPU usage before generation
    print_gpu_usage("(Before Image Generation)")
    
    # Create and start real-time GPU monitor
    gpu_monitor = GPUMonitor(interval=1.5)  # Check every 1.5 seconds
    
    try:
        # Start real-time monitoring
        gpu_monitor.start()
        
        # Generate image with some optimization parameters
        images = pipe(
            prompt=prompt,
            num_inference_steps=20,  # Good balance of quality vs speed
            guidance_scale=7.5,      # Standard guidance scale
            height=1024,             # Reduced for memory efficiency
            width=1024               # Reduced for memory efficiency
        ).images[0]
        
        # Stop monitoring
        gpu_monitor.stop()
        
        # Check GPU usage after generation
        print_gpu_usage("(After Image Generation)")
        
        return images
        
    except Exception as e:
        # Stop monitoring on error
        gpu_monitor.stop()
        print(f"❌ Error generating image: {e}")
        print_gpu_usage("(After Generation Error)")
        return None

def save_image(image, prompt):
    """Save the generated image to file."""
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.png"
    
    try:
        image.save(filename)
        print(f"💾 Image saved as: {filename}")
        print(f"📍 Full path: {os.path.abspath(filename)}")
        
        # Also save as output.png for easy access
        image.save("output.png")
        print(f"💾 Also saved as: output.png")
        
        return filename
        
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return None

def main():
    """Main function to orchestrate the image generation process."""
    print("🔥 Starting Stable Diffusion XL Image Generation")
    print("="*60)
    
    # Check GPU availability
    if not check_gpu_availability():
        sys.exit(1)
    
    # Initial GPU usage baseline
    print_gpu_usage("(Initial Baseline)")
    
    # Load model
    pipe = load_model()
    if pipe is None:
        sys.exit(1)
    
    # Get prompt from user
    prompt = get_prompt()
    
    # Generate image
    image = generate_image(pipe, prompt)
    if image is None:
        sys.exit(1)
    
    # Save image
    filename = save_image(image, prompt)
    if filename is None:
        sys.exit(1)
    
    print("\n🎉 Image generation completed successfully!")
    print(f"📝 Prompt: {prompt}")
    print(f"🖼️  Output: {filename}")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    print("🧹 GPU cache cleared")

if __name__ == "__main__":
    try:
        main()
        torch.cuda.empty_cache()
        
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
        torch.cuda.empty_cache()
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        torch.cuda.empty_cache()
        sys.exit(1) 