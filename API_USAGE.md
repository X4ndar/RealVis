# RealVis 5 API Client - External API Proxy

This is the **RealVis 5 API Client**, a proxy service that forwards requests to your external RealVis 5 API hosted on RunPod. Designed for seamless integration with your Next.js frontend while providing authentication, CORS, and request management features.

## 🔄 Architecture Overview

```
Next.js Frontend → API Client (this app) → External RunPod API
```

- **Frontend**: Your Next.js application
- **API Client**: This proxy service (provides auth, CORS, request forwarding)
- **External API**: Your RunPod instance at `https://do9n3s330iext0-8000.proxy.runpod.net/`

## 🚀 Quick Start

1. **Create your `.env` file** (copy from `text.txt` and rename to `.env`)
2. **Configure the external API URL** in your `.env` file
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run the proxy**: `python app.py`

## 🔧 Configuration

### External API Configuration
```env
# Your RunPod API endpoint
API_BASE_URL=https://do9n3s330iext0-8000.proxy.runpod.net

# Timeout for external API requests (seconds)
EXTERNAL_API_TIMEOUT=300
```

### Authentication & Security
```env
# Optional API key for your proxy
API_KEY=your-secure-api-key-here

# Leave empty to disable authentication
API_KEY=
```

### Server Configuration
```env
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=production
DEBUG=false
```

### CORS (for Frontend Integration)
```env
# Allow your frontend domain
ALLOWED_ORIGINS=https://your-frontend-domain.com

# Multiple origins (comma-separated)
ALLOWED_ORIGINS=https://yourdomain.com,http://localhost:3000

# Allow all (development only)
ALLOWED_ORIGINS=*
```

## 📡 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "external_api_url": "https://do9n3s330iext0-8000.proxy.runpod.net",
  "external_api_status": "healthy",
  "api_version": "3.1.0",
  "environment": "production",
  "mode": "client_proxy"
}
```

### 2. Text-to-Image Generation (Proxied)
```http
POST /generate
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY  # If authentication enabled
```

**Request Body:**
```json
{
  "prompt": "a stunning portrait of a beautiful woman with flowing hair, natural lighting, photorealistic",
  "negative_prompt": "cartoon, anime, painted, artificial, low quality",
  "num_inference_steps": 25,
  "guidance_scale": 2.0,
  "height": 1024,
  "width": 1024
}
```

**Response:**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAA...",  // Base64 encoded image
  "prompt": "a stunning portrait of a beautiful woman...",
  "negative_prompt": "cartoon, anime, painted, artificial...",
  "generation_time": 12.34,
  "model_used": "external_api"
}
```

### 3. Image Inpainting (Proxied)
```http
POST /inpaint
Content-Type: multipart/form-data
Authorization: Bearer YOUR_API_KEY  # If authentication enabled
```

**Form Fields:**
- `image`: (file) Original image
- `mask`: (file) Black/white mask (white = areas to regenerate)
- `prompt`: (text) Description of what to generate
- `negative_prompt`: (text, optional) What to avoid
- `num_inference_steps`: (int, default: 30)
- `guidance_scale`: (float, default: 8.0)
- `strength`: (float, default: 0.95)

**Response:**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAA...",  // Base64 encoded result
  "prompt": "a red baseball cap",
  "negative_prompt": "cartoon, anime, painted...",
  "generation_time": 8.76,
  "model_used": "external_api"
}
```

## 🌐 Frontend Integration Examples

### JavaScript/TypeScript (Next.js)

```javascript
// Configure your API client endpoint (this proxy)
const API_CLIENT_URL = 'http://your-proxy-host:8000';

// Text-to-image generation via proxy
async function generateImage(prompt) {
  const response = await fetch(`${API_CLIENT_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer YOUR_PROXY_API_KEY', // If using proxy authentication
    },
    body: JSON.stringify({
      prompt: prompt,
      num_inference_steps: 25,
      guidance_scale: 2.0,
      height: 1024,
      width: 1024
    })
  });

  if (!response.ok) {
    throw new Error(`Proxy API Error: ${response.status}`);
  }

  const result = await response.json();
  
  // Display the image
  const imageElement = document.createElement('img');
  imageElement.src = `data:image/png;base64,${result.image}`;
  document.body.appendChild(imageElement);
  
  return result;
}

// Inpainting example via proxy
async function inpaintImage(imageFile, maskFile, prompt) {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('mask', maskFile);
  formData.append('prompt', prompt);
  formData.append('num_inference_steps', '30');
  formData.append('guidance_scale', '8.0');
  formData.append('strength', '0.95');

  const response = await fetch(`${API_CLIENT_URL}/inpaint`, {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer YOUR_PROXY_API_KEY', // If using proxy authentication
    },
    body: formData
  });

  const result = await response.json();
  return result;
}

// Check proxy and external API health
async function checkHealth() {
  const response = await fetch(`${API_CLIENT_URL}/health`);
  const health = await response.json();
  
  console.log('Proxy Status:', health.status);
  console.log('External API Status:', health.external_api_status);
  console.log('External API URL:', health.external_api_url);
  
  return health;
}
```

### Python Client Example

```python
import requests
import base64
from PIL import Image
from io import BytesIO

class RealVisProxyClient:
    def __init__(self, proxy_url, api_key=None):
        self.proxy_url = proxy_url.rstrip('/')
        self.api_key = api_key
        
    def _get_headers(self):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    def generate_image(self, prompt, **kwargs):
        """Generate image via proxy."""
        data = {'prompt': prompt, **kwargs}
        
        response = requests.post(
            f'{self.proxy_url}/generate',
            json=data,
            headers=self._get_headers()
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Decode base64 image
        image_data = base64.b64decode(result['image'])
        image = Image.open(BytesIO(image_data))
        
        return image, result
    
    def health_check(self):
        """Check proxy and external API health."""
        response = requests.get(f'{self.proxy_url}/health')
        return response.json()

# Usage
proxy_client = RealVisProxyClient('http://your-proxy-host:8000', 'YOUR_PROXY_API_KEY')

# Check health
health = proxy_client.health_check()
print(f"Proxy: {health['status']}")
print(f"External API: {health['external_api_status']}")

# Generate an image
image, metadata = proxy_client.generate_image(
    prompt="a beautiful landscape with mountains and a lake",
    num_inference_steps=25,
    guidance_scale=2.0
)

image.save('generated_image.png')
print(f"Generated via external API in {metadata['generation_time']} seconds")
```

## 🔐 Authentication

The proxy supports **two levels of authentication**:

1. **Proxy Authentication** (optional): Protect access to your proxy
   ```env
   API_KEY=your-secure-proxy-key
   ```

2. **External API Authentication**: Configured in the external RunPod instance

## ⚡ Benefits of Using the Proxy

- **🔐 Security**: Add authentication layer to your external API
- **🌍 CORS Management**: Handle cross-origin requests for your frontend
- **📊 Monitoring**: Monitor external API health and requests
- **🔄 Request Management**: Centralized request handling and error management
- **⚙️ Configuration**: Environment-based settings for different deployments
- **🛡️ Error Handling**: Enhanced error messages and status codes

## 🚨 Error Handling

The proxy provides detailed error responses:

```json
{
  "detail": "External API timeout after 300 seconds"
}
```

Common error codes:
- **503**: External API unreachable
- **504**: External API timeout
- **401**: Proxy authentication required
- **4xx/5xx**: External API errors (forwarded)

## 🔧 Development vs Production

### Development Setup
```env
API_KEY=                           # No proxy authentication
ALLOWED_ORIGINS=*                  # Allow all origins
DEBUG=true                         # Enable hot reload
LOG_LEVEL=DEBUG                    # Verbose logging
API_BASE_URL=https://do9n3s330iext0-8000.proxy.runpod.net
```

### Production Setup
```env
API_KEY=super-secure-proxy-key     # Enable proxy authentication
ALLOWED_ORIGINS=https://yourdomain.com  # Restrict origins
DEBUG=false                        # Production mode
LOG_LEVEL=INFO                     # Standard logging
API_BASE_URL=https://do9n3s330iext0-8000.proxy.runpod.net
```

## 📊 Monitoring & Debugging

- **Health Endpoint**: `/health` - Check proxy and external API status
- **Interactive Docs**: `/docs` - Test the proxy endpoints
- **External API Monitoring**: Real-time status of your RunPod instance
- **Request Logging**: Track all forwarded requests and responses

## 🔄 Request Flow

1. **Frontend** → makes request to **Proxy** (`/generate` or `/inpaint`)
2. **Proxy** → validates request, applies authentication/CORS
3. **Proxy** → forwards request to **External API** (RunPod)
4. **External API** → processes request, generates image
5. **External API** → returns response to **Proxy**
6. **Proxy** → forwards response to **Frontend**

## 🚀 Deployment Tips

### RunPod + Proxy Setup
- **External API**: Your RunPod instance with the actual models
- **Proxy**: This client app (can run anywhere - local, cloud, etc.)
- **Frontend**: Your Next.js app connects to the proxy

### Benefits of This Architecture
- **Scalability**: Scale proxy and external API independently
- **Security**: Add authentication without modifying external API
- **Flexibility**: Easy to switch external API providers
- **Development**: Test locally while using production external API

---

Your RealVis 5 API Client is now ready to proxy requests to your external RunPod API! 🎨✨ 