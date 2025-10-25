# Face Recognition System - REST API Documentation

## Base URL
```
http://localhost:5000/api/v1
```

## Authentication
Currently, no authentication is required. In production, implement API keys or JWT tokens.

---

## Endpoints

### 1. Health Check
**GET** `/api/v1/health`

Check system health and database connectivity.

**Response (200):**
```json
{
  "status": "healthy",
  "database": "connected",
  "total_users": 5
}
```

---

### 2. Get All Users
**GET** `/api/v1/users`

Retrieve all registered users.

**Response (200):**
```json
{
  "status": "success",
  "data": [
    {
      "id": 1,
      "name": "John Doe",
      "email": null,
      "created_at": "2024-10-25T10:00:00",
      "is_active": true
    },
    {
      "id": 2,
      "name": "Jane Smith",
      "email": null,
      "created_at": "2024-10-25T10:05:00",
      "is_active": true
    }
  ]
}
```

---

### 3. Get User by Name
**GET** `/api/v1/users/<name>`

Retrieve specific user details.

**Parameters:**
- `name` (string, path): User name

**Response (200):**
```json
{
  "status": "success",
  "data": {
    "id": 1,
    "name": "John Doe",
    "email": null,
    "created_at": "2024-10-25T10:00:00",
    "is_active": true
  }
}
```

**Response (404):**
```json
{
  "status": "error",
  "message": "User not found"
}
```

---

### 4. Register New User
**POST** `/api/v1/register`

Register a new user with face encoding.

**Request (multipart/form-data):**
- `name` (string): User's name
- `file` (file): Face image (JPG, PNG, etc.)

**cURL Example:**
```bash
curl -X POST http://localhost:5000/api/v1/register \
  -F "name=John Doe" \
  -F "file=@/path/to/image.jpg"
```

**Response (201 - Success):**
```json
{
  "status": "success",
  "message": "User John Doe registered successfully",
  "user": {
    "id": 1,
    "name": "John Doe",
    "email": null,
    "created_at": "2024-10-25T10:00:00",
    "is_active": true
  }
}
```

**Response (400 - Error):**
```json
{
  "status": "error",
  "message": "No face found in image"
}
```

**Error Messages:**
- "Missing file or name" - File or name parameter missing
- "Name is required" - Name is empty
- "No face found in image" - No face detected
- "Multiple faces detected" - More than one face in image
- "User already exists" - User name already registered

---

### 5. Verify Face
**POST** `/api/v1/verify`

Verify a face image against registered users.

**Request (multipart/form-data):**
- `image` (file): Face image to verify

**cURL Example:**
```bash
curl -X POST http://localhost:5000/api/v1/verify \
  -F "image=@/path/to/test_image.jpg"
```

**Response (200 - Verified):**
```json
{
  "status": "success",
  "data": {
    "status": "verified",
    "message": "Verified as John Doe",
    "user_id": 1,
    "name": "John Doe"
  }
}
```

**Response (404 - Unknown):**
```json
{
  "status": "unknown",
  "message": "Unknown face"
}
```

**Response (400 - Error):**
```json
{
  "status": "error",
  "message": "No face detected"
}
```

---

### 6. Check Duplicate
**POST** `/api/v1/check-duplicate`

Check if a face is a duplicate of existing users.

**Request (multipart/form-data):**
- `image` (file): Face image to check

**cURL Example:**
```bash
curl -X POST http://localhost:5000/api/v1/check-duplicate \
  -F "image=@/path/to/image.jpg"
```

**Response (200 - No Duplicate):**
```json
{
  "duplicate": false,
  "best_distance": 0.87
}
```

**Response (200 - Duplicate Found):**
```json
{
  "duplicate": true,
  "matches": ["John Doe", "Jane Smith"],
  "best_distance": 0.35
}
```

**Response (200 - Special Cases):**
```json
{
  "status": "no_face"
}
```

```json
{
  "status": "multiple_faces"
}
```

---

### 7. Get Verification Logs
**GET** `/api/v1/verification-logs`

Retrieve verification attempt logs.

**Query Parameters:**
- `limit` (integer, optional): Number of logs to retrieve (default: 100)

**Example:**
```
GET /api/v1/verification-logs?limit=50
```

**Response (200):**
```json
{
  "status": "success",
  "data": [
    {
      "id": 1,
      "user_id": 1,
      "status": "verified",
      "message": "Verified as John Doe",
      "created_at": "2024-10-25T10:30:00"
    },
    {
      "id": 2,
      "user_id": null,
      "status": "unknown",
      "message": "Unknown face",
      "created_at": "2024-10-25T10:31:00"
    }
  ]
}
```

---

### 8. Get All Encodings
**GET** `/api/v1/encodings`

Get all face encodings for client-side processing.

**Response (200):**
```json
{
  "status": "success",
  "encodings": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...]
  ],
  "names": ["John Doe", "Jane Smith"]
}
```

---

## Error Handling

All errors follow this format:

```json
{
  "status": "error",
  "message": "Error description"
}
```

### HTTP Status Codes
- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

---

## Database Schema

### Users Table
```sql
CREATE TABLE users (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  is_active BOOLEAN DEFAULT TRUE,
  INDEX idx_name (name)
);
```

### Face Encodings Table
```sql
CREATE TABLE face_encodings (
  id INT PRIMARY KEY AUTO_INCREMENT,
  user_id INT NOT NULL,
  encoding LONGTEXT NOT NULL,
  image_path VARCHAR(500),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  is_active BOOLEAN DEFAULT TRUE,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_user_id (user_id)
);
```

### Verification Logs Table
```sql
CREATE TABLE verification_logs (
  id INT PRIMARY KEY AUTO_INCREMENT,
  user_id INT,
  status VARCHAR(50) NOT NULL,
  message VARCHAR(500),
  image_path VARCHAR(500),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
  INDEX idx_user_id (user_id),
  INDEX idx_created_at (created_at)
);
```

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Database
Edit `.env` file:
```env
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=root
MYSQL_DATABASE=face_recognition_db
MYSQL_PORT=3306
FLASK_SECRET=your-secret-key
DEBUG=False
```

### 3. Create Database
```bash
mysql -u root -p -e "CREATE DATABASE face_recognition_db;"
```

### 4. Run Application
```bash
python app.py
```

The API will be available at `http://localhost:5000/api/v1`

---

## Example Client Usage (Python)

```python
import requests

API_URL = "http://localhost:5000/api/v1"

# Register user
files = {'file': open('user_photo.jpg', 'rb')}
data = {'name': 'John Doe'}
response = requests.post(f"{API_URL}/register", files=files, data=data)
print(response.json())

# Verify face
files = {'image': open('test_photo.jpg', 'rb')}
response = requests.post(f"{API_URL}/verify", files=files)
print(response.json())

# Get all users
response = requests.get(f"{API_URL}/users")
print(response.json())

# Check health
response = requests.get(f"{API_URL}/health")
print(response.json())
```

---

## Support
For issues or questions, contact: support@example.com
