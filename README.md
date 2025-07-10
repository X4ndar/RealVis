# 🎨 RealVis 5 API Client - External API Proxy

> **Proxy service that forwards requests to your external RealVis 5 API on RunPod**

This repository contains a FastAPI-based proxy/client that forwards image generation requests to your external RealVis 5 API hosted on RunPod. It provides authentication, CORS management, and seamless integration for your Next.js frontend.

## 🔄 Architecture

```
Next.js Frontend → API Client (this app) → External RunPod API
                     ↓
              Auth + CORS + Monitoring
```

## ✨ Features

- 🔄 **Request Proxy**: Forward requests to external RealVis 5 API
- 🔐 **Optional Authentication**: Secure proxy access with API keys
- 🌍 **CORS Management**: Handle cross-origin requests for frontends
- 📊 **Health Monitoring**: Monitor external API status and availability
- ⚡ **Request Management**: Centralized error handling and timeouts
- 🔧 **Environment Configuration**: Flexible deployment settings
- 📝 **Comprehensive Logging**: Track all proxy requests and responses

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to the project
cd your-project-directory

# Create .env file from the example in text.txt
cp text.txt .env

# Edit .env with your external API URL
nano .env
```

### 2. Configure External API

Edit your `.env` file:

```env
# Your RunPod API endpoint
API_BASE_URL=https://do9n3s330iext0-8000.proxy.runpod.net

# Optional proxy authentication
API_KEY=your-secure-proxy-key

# CORS for your frontend
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Proxy

```bash
python app.py
```

Your proxy will be available at `http://0.0.0.0:8000`

## 📡 Proxy Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Proxy and external API health status |
| `/generate` | POST | Proxy text-to-image requests |
| `/inpaint` | POST | Proxy inpainting requests |
| `/docs` | GET | Interactive API documentation |
| `/api` | GET | Proxy configuration info |

## 🔧 Configuration

### External API Settings
```env
# Your external RunPod API
API_BASE_URL=https://do9n3s330iext0-8000.proxy.runpod.net
EXTERNAL_API_TIMEOUT=300

# Proxy server
HOST=0.0.0.0
PORT=8000
```

### Authentication (Optional)
```env
# Enable proxy authentication
API_KEY=your-secure-proxy-key

# Disable authentication (development)
API_KEY=
```

### CORS Configuration
```env
# Production: Specific domain
ALLOWED_ORIGINS=https://yourdomain.com

# Development: Allow all
ALLOWED_ORIGINS=*

# Multiple domains
ALLOWED_ORIGINS=https://domain1.com,https://domain2.com
```

## 🌐 Frontend Integration

### JavaScript/Next.js Example

```javascript
// Connect to your proxy (not directly to RunPod)
const PROXY_URL = 'http://your-proxy-host:8000';

const response = await fetch(`${PROXY_URL}/generate`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_PROXY_API_KEY' // If proxy auth enabled
  },
  body: JSON.stringify({
    prompt: "a stunning photorealistic portrait",
    num_inference_steps: 25,
    guidance_scale: 2.0,
    height: 1024,
    width: 1024
  })
});

const result = await response.json();
// result.image contains base64 encoded image
```

### Health Check Example

```javascript
// Check both proxy and external API status
const healthResponse = await fetch(`${PROXY_URL}/health`);
const health = await healthResponse.json();

console.log('Proxy Status:', health.status);
console.log('External API Status:', health.external_api_status);
console.log('External API URL:', health.external_api_url);
```

## 🔐 Authentication Layers

The proxy supports **two authentication levels**:

1. **Proxy Authentication** (optional):
   - Protects access to your proxy service
   - Set `API_KEY` in proxy's `.env`
   - Include `Authorization: Bearer PROXY_KEY` in requests

2. **External API Authentication**:
   - Handled by your RunPod instance
   - Configured on the external API side

## ⚡ Benefits of Proxy Architecture

### 🔐 Security
- Add authentication without modifying external API
- Protect direct access to RunPod endpoints
- Centralized access control

### 🌍 CORS Management
- Handle cross-origin requests for web frontends
- Avoid CORS issues between frontend and RunPod
- Flexible origin configuration

### 📊 Monitoring & Control
- Monitor external API health and availability
- Track request/response patterns
- Centralized error handling and logging

### 🔄 Flexibility
- Easy to switch external API providers
- Independent scaling of proxy and external API
- Development/testing with different external APIs

## 🚨 Error Handling

The proxy provides enhanced error responses:

```json
{
  "detail": "External API timeout after 300 seconds"
}
```

**Common Error Codes:**
- `503` - External API unreachable
- `504` - External API timeout  
- `401` - Proxy authentication required
- `4xx/5xx` - External API errors (forwarded)

## 🔧 Deployment Scenarios

### Development Setup
```env
API_KEY=                    # No proxy auth
ALLOWED_ORIGINS=*           # Allow all origins  
DEBUG=true                  # Hot reload
LOG_LEVEL=DEBUG            # Verbose logging
API_BASE_URL=https://do9n3s330iext0-8000.proxy.runpod.net
```

### Production Setup
```env
API_KEY=super-secure-key    # Enable proxy auth
ALLOWED_ORIGINS=https://yourdomain.com  # Restrict origins
DEBUG=false                 # Production mode
LOG_LEVEL=INFO             # Standard logging
API_BASE_URL=https://do9n3s330iext0-8000.proxy.runpod.net
```

### Multi-Environment
- **Local Proxy**: Run proxy locally for development
- **Cloud Proxy**: Deploy proxy to cloud for production
- **External API**: RunPod instance remains the same

## 📊 Monitoring

### Health Monitoring
```bash
curl http://your-proxy:8000/health
```

Response includes:
- Proxy service status
- External API connectivity
- External API health status
- Configuration details

### Request Logging
The proxy logs all:
- Incoming requests from frontend
- Outgoing requests to external API
- Response times and status codes
- Error conditions and timeouts

## 🚀 Deployment Options

### Local Development
```bash
# Run proxy locally
python app.py

# Frontend connects to localhost:8000
# Proxy forwards to RunPod
```

### Cloud Deployment
- Deploy proxy to any cloud provider
- Configure `API_BASE_URL` to your RunPod instance
- Frontend connects to cloud proxy
- Benefits: Better uptime, global availability

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## 🛠 Development

The proxy is built with:
- **FastAPI**: Modern Python web framework
- **httpx**: Async HTTP client for external API calls
- **Pydantic**: Data validation and serialization
- **Environment Configuration**: Flexible `.env` setup

## 📚 Documentation

- **API Usage**: See `API_USAGE.md` for detailed integration guide
- **Configuration**: Complete `.env` example in `text.txt`
- **Interactive Docs**: Available at `/docs` when running
- **External API**: Your RunPod instance documentation

## 🔄 Request Flow

1. **Frontend** sends request to **Proxy**
2. **Proxy** validates authentication & CORS
3. **Proxy** forwards request to **External API** (RunPod)
4. **External API** processes and returns response
5. **Proxy** forwards response back to **Frontend**

## 🤝 Support

For questions about:
- **Proxy Configuration**: Check `text.txt` for environment options
- **Integration**: See examples in `API_USAGE.md`  
- **External API**: Refer to your RunPod instance documentation
- **Testing**: Use interactive docs at `/docs`

---

**Ready to proxy requests to your RunPod RealVis 5 API!** 🚀✨ 