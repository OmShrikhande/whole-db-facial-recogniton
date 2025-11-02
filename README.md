# Person Verification API

A facial recognition and person verification REST API built with Node.js, Express, and face-api.js. This backend service enables face detection, recognition, and registration for verification systems.

## Features

- **Face Detection & Analysis**: Advanced face detection with quality validation (brightness, blur, size)
- **Face Registration**: Register new faces with unique user codes
- **Face Verification**: Verify faces against registered users with configurable thresholds
- **Duplicate Detection**: Prevent duplicate face registrations
- **RESTful API**: Clean REST endpoints for integration
- **Production Ready**: Environment-based configuration, health checks, and monitoring
- **MongoDB Storage**: Scalable document storage for face embeddings
- **Docker Support**: Containerized deployment with Docker Compose

## Requirements

- Node.js 16+
- MongoDB (local or cloud)
- The following npm packages (see `package.json`):
  - `express` - Web framework
  - `face-api.js` - Face recognition library
  - `@tensorflow/tfjs` - Machine learning framework
  - `mongodb` - Database driver

## Quick Start

1. **Clone and install dependencies**:
   ```bash
   git clone <repository-url>
   cd person-verification
   npm install
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB connection and other settings
   ```

3. **Start MongoDB** (if running locally):
   ```bash
   mongod
   ```

4. **Start the API server**:
   ```bash
   npm run dev  # Development mode with auto-reload
   # or
   npm start    # Production mode
   ```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health & Status
- `GET /health` - Health check endpoint
- `GET /status` - System status and configuration
- `GET /metrics` - Performance metrics and statistics

### Face Operations
- `POST /verify` - Verify a face against registered users
- `POST /register` - Register a new face (legacy endpoint)

### Request/Response Examples

#### Verify Face
```bash
curl -X POST http://localhost:5000/verify \
  -F "image=@face_image.jpg" \
  -F "usercode=user123"
```

**Success Response (accept):**
```json
{
  "status": "accept",
  "name": "John Doe",
  "message": "Accepted — good luck"
}
```

**Success Response (registered):**
```json
{
  "status": "registered",
  "message": "Face registered successfully for verification"
}
```

**Error Response (duplicate face):**
```json
{
  "status": "duplicate-face",
  "message": "This face is already registered with usercode:",
  "existingUsercode": "existing123",
  "similarity": "95%"
}
```

**Error Response (reupload):**
```json
{
  "status": "reupload",
  "reason": "too-blurry",
  "message": "Image is blurry - please hold the camera steady"
}
```

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and modify as needed:

### Database
- `MONGODB_URL` - MongoDB connection string
- `MONGODB_DATABASE` - Database name (default: faceDB)

### Server
- `PORT` - Server port (default: 5000)
- `HOST` - Server host (default: 0.0.0.0)
- `NODE_ENV` - Environment (development/production)

### Face Recognition
- `DUPLICATE_THRESHOLD` - Threshold for duplicate detection (default: 0.45)
- `VERIFICATION_THRESHOLD` - Threshold for face matching (default: 0.45)
- `MAX_FILE_SIZE_MB` - Maximum upload file size in MB (default: 25)

### Security
- `CORS_ORIGINS` - Comma-separated list of allowed origins
- `JWT_SECRET` - Secret key for JWT tokens (if used)

## Deployment

### Development
```bash
npm run dev  # Uses nodemon for auto-reload
```

### Production
```bash
npm run prod  # Production mode
```

### PM2 (Process Manager)
```bash
npm run pm2:start   # Start with PM2
npm run pm2:logs    # View logs
npm run pm2:monit   # Monitor processes
```

### Docker
```bash
# Build and run with Docker
npm run docker:build
npm run docker:run

# Or use Docker Compose
npm run docker:up
npm run docker:down
```

## Project Structure

```
├── server.js              # Main Express application
├── package.json           # Dependencies and scripts
├── .env                   # Environment configuration
├── models/                # Face recognition models
├── uploads/               # Temporary file uploads
├── encodings.pkl          # Legacy encodings (if any)
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── ecosystem.config.js    # PM2 configuration
└── README.md             # This file
```

## File Descriptions

- **server.js**: Main Express application with face recognition logic
- **models/**: Pre-trained face detection and recognition models
- **.env**: Environment configuration (not committed to git)
- **package.json**: Node.js dependencies and npm scripts
- **Dockerfile**: Container build configuration
- **ecosystem.config.js**: PM2 process manager configuration

## Face Quality Validation

The API performs comprehensive image quality checks:

- **Face Detection**: Ensures at least one face is present
- **Face Size**: Minimum 80x80 pixels
- **Detection Confidence**: Minimum 50% confidence
- **Brightness**: Optimal luminance range (40-230)
- **Blur Detection**: Variance of Laplacian > 100
- **Multiple Faces**: Rejects images with multiple faces

## Database Schema

### Collections

#### image_verifications
Stores face embeddings and verification data:
```javascript
{
  usercode: "string",
  imagetext: [array of floats], // Face embedding
  status: "registered" | "rejected",
  rejectionReason: "duplicate-face",
  matchedUsercode: "string", // For duplicates
  similarity: number,
  createdAt: Date
}
```

#### faces (legacy)
Simple face storage for registered users:
```javascript
{
  name: "string",
  embedding: [array of floats]
}
```

## Monitoring

The API provides several monitoring endpoints:

- **Health Check**: `GET /health` - Basic health status
- **System Status**: `GET /status` - Detailed system information
- **Metrics**: `GET /metrics` - Performance and usage statistics

## Security Considerations

- File upload size limits
- CORS policy enforcement
- Input validation for user codes
- Secure environment variable handling
- No sensitive data in logs

## Troubleshooting

### Common Issues

1. **"No face detected"**: Ensure good lighting and clear face visibility
2. **"Image too dark/bright"**: Adjust lighting conditions
3. **"Multiple faces detected"**: Ensure only one person in frame
4. **"Face too small"**: Move closer to camera

### Logs
Check application logs for detailed error information:
```bash
npm run pm2:logs  # If using PM2
# or check console output in development
```

## License

MIT License
