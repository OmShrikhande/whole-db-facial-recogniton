# Frontend API Configuration Guide

## Overview

The frontend is now **fully configurable** to point to any hosted backend using the `static/config.js` file.

---

## Configuration File Location

**File:** `static/config.js`

This file is automatically loaded by all HTML templates and contains the API base URL.

---

## How to Configure

### Step 1: Open `static/config.js`

```javascript
// For local development (backend and frontend on same server)
const API_BASE_URL = window.location.origin;

// For production with separate backend server, use:
// const API_BASE_URL = 'http://your-backend-domain.com';
// const API_BASE_URL = 'https://your-backend-domain.com';
// const API_BASE_URL = 'http://192.168.1.100:5000';
```

### Step 2: Change the URL Based on Your Deployment

#### Local Development (Same Server)
```javascript
const API_BASE_URL = window.location.origin;
```
**Use when:** Frontend and backend run on the same server  
**Example:** Both served from `http://localhost:5000`

---

#### Remote Backend (Different Server - HTTP)
```javascript
const API_BASE_URL = 'http://your-backend-domain.com';
```
**Use when:** Backend is on a different server  
**Examples:**
- `http://api.example.com`
- `http://192.168.1.100:5000`
- `http://backend-server.local:5000`

---

#### Remote Backend (HTTPS - Production)
```javascript
const API_BASE_URL = 'https://your-backend-domain.com';
```
**Use when:** Backend has SSL/HTTPS enabled  
**Examples:**
- `https://api.example.com`
- `https://secure-backend.example.com:8443`

---

## Common Deployment Scenarios

### Scenario 1: Everything on Localhost (Development)
```javascript
const API_BASE_URL = window.location.origin;
// Automatically becomes: http://localhost:5000
```

### Scenario 2: Frontend on Localhost, Backend on Remote Server
```javascript
const API_BASE_URL = 'http://192.168.1.50:5000';
// Or for domain:
const API_BASE_URL = 'http://backend-server.example.com';
```

### Scenario 3: Both on Production Server with Domain
```javascript
const API_BASE_URL = 'https://api.mycompany.com';
```

### Scenario 4: Frontend and Backend on Different Ports (Dev)
```javascript
const API_BASE_URL = 'http://localhost:5000';
// Frontend runs on port 3000 or 8080
// Backend runs on port 5000
```

### Scenario 5: Docker/Container Environment
```javascript
const API_BASE_URL = 'http://backend-service:5000';
// When using Docker Compose networking
```

---

## API Endpoints Available

After configuring the base URL, the frontend automatically has access to these endpoints:

```javascript
API_ENDPOINTS.HEALTH              // /api/v1/health
API_ENDPOINTS.USERS               // /api/v1/users
API_ENDPOINTS.REGISTER            // /api/v1/register
API_ENDPOINTS.VERIFY              // /api/v1/verify
API_ENDPOINTS.CHECK_DUPLICATE     // /api/v1/check-duplicate
API_ENDPOINTS.VERIFICATION_LOGS   // /api/v1/verification-logs
API_ENDPOINTS.ENCODINGS           // /api/v1/encodings

// Legacy Flask routes
API_ENDPOINTS.VIDEO_FEED          // /video_feed
API_ENDPOINTS.VERIFICATION_STATUS // /verification_status
API_ENDPOINTS.STOP_VERIFICATION   // /stop_verification
```

---

## Testing Configuration

### Check if Configuration is Loaded

Open browser console (F12) and type:
```javascript
console.log(API_BASE_URL);
console.log(API_ENDPOINTS);
```

You should see the configured base URL and all available endpoints.

### Test API Connectivity

```javascript
// Test health check
fetch(API_ENDPOINTS.HEALTH)
  .then(r => r.json())
  .then(d => console.log('✓ Backend connected:', d))
  .catch(e => console.error('✗ Backend error:', e));
```

---

## CORS Issues (Cross-Origin)

### Problem
Frontend and backend are on different domains/ports and you get CORS errors.

### Solution 1: Use Same Domain (Recommended)
Put both frontend and backend behind Nginx or Apache with same domain.

### Solution 2: Enable CORS on Backend
Add CORS headers to Flask app:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
```

Then install:
```bash
pip install flask-cors
```

### Solution 3: Use HTTPS
Both frontend and backend must use HTTPS for CORS to work properly in production.

---

## Environment-Specific Configuration

### Development (.env for Frontend)
If using a frontend build process, create `.env.development`:
```
REACT_APP_API_URL=http://localhost:5000
```

### Production (.env for Frontend)
Create `.env.production`:
```
REACT_APP_API_URL=https://api.mycompany.com
```

Then in `config.js`:
```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://api.mycompany.com';
```

---

## Deployment Examples

### Example 1: Deploying Frontend + Backend on Same Nginx Server

**`nginx.conf`:**
```nginx
upstream backend {
    server localhost:5000;
}

server {
    listen 80;
    server_name myapp.com;

    # Serve frontend
    location / {
        root /var/www/face-recognition/templates;
        try_files $uri $uri/ =404;
    }

    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://backend;
    }

    # Proxy other backend routes
    location /video_feed {
        proxy_pass http://backend;
    }
}
```

**Frontend `config.js`:**
```javascript
const API_BASE_URL = window.location.origin;
// Becomes: https://myapp.com
```

---

### Example 2: Separate Frontend (CDN) + Backend (AWS/Azure)

**Frontend deployed to:** `https://myapp.netlify.com`  
**Backend deployed to:** `https://api.myapp.com`

**Frontend `config.js`:**
```javascript
const API_BASE_URL = 'https://api.myapp.com';
```

**Backend CORS:**
```python
CORS(app, origins=['https://myapp.netlify.com'])
```

---

### Example 3: Docker Compose Setup

**`docker-compose.yml`:**
```yaml
version: '3'
services:
  backend:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MYSQL_HOST=db
      - FLASK_ENV=production

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  db:
    image: mysql:8
    environment:
      - MYSQL_ROOT_PASSWORD=root
```

**Frontend `config.js`:**
```javascript
const API_BASE_URL = 'http://backend:5000';
// Or from environment variable
const API_BASE_URL = process.env.API_BASE_URL || 'http://backend:5000';
```

---

## Troubleshooting

### Issue: API calls return 404

**Cause:** Wrong base URL configured  
**Solution:** Check `config.js` - ensure URL is correct

**Test:**
```javascript
console.log(API_ENDPOINTS.HEALTH);
// Should show: http://your-backend.com/api/v1/health
```

---

### Issue: CORS Error in Browser Console

**Error:** `Access to XMLHttpRequest has been blocked by CORS policy`

**Solution:**
1. Ensure backend has CORS enabled
2. Both URLs use same protocol (http or https)
3. If different domains, add to backend:
```python
from flask_cors import CORS
CORS(app)
```

---

### Issue: Video stream not loading in camera.html

**Cause:** Video feed endpoint URL is wrong  
**Solution:** Check that `API_ENDPOINTS.VIDEO_FEED` is correct:
```javascript
console.log(API_ENDPOINTS.VIDEO_FEED);
// Should be: http://your-backend.com/video_feed
```

---

### Issue: Connection refused / Cannot reach backend

**Cause:** Backend not running or wrong address  
**Solution:** 
1. Verify backend is running: `python app.py`
2. Test URL in browser: `http://backend-url/api/v1/health`
3. Check firewall/network connectivity

---

## Security Recommendations

✅ **Use HTTPS in Production**
```javascript
const API_BASE_URL = 'https://api.mycompany.com';
```

✅ **Enable CORS Correctly**
Only allow your frontend domain in backend CORS config

✅ **Keep Sensitive Data Secure**
Never commit sensitive URLs in `config.js` - use environment variables

✅ **Use API Authentication**
Consider adding JWT or API keys for production

---

## Summary

| Deployment Type | Config |
|---|---|
| **Local Development** | `window.location.origin` |
| **Same Server** | `window.location.origin` |
| **Different Server (HTTP)** | `http://backend-domain:5000` |
| **Different Server (HTTPS)** | `https://backend-domain` |
| **Docker Network** | `http://service-name:5000` |
| **Production** | `https://api.yourdomain.com` |

---

## Next Steps

1. **Edit** `static/config.js` with your backend URL
2. **Test** API connectivity from browser console
3. **Deploy** frontend and backend
4. **Monitor** browser console for any errors
5. **Enable CORS** if frontend and backend are separate

---

**Configuration ready! Your frontend can now connect to any backend.** 🚀
