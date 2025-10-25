// ============================================================================
// FRONTEND API CONFIGURATION
// Change API_BASE_URL to point to your hosted backend
// ============================================================================

// For local development (backend and frontend on same server)
const API_BASE_URL = window.location.origin;

// For production with separate backend server, use:
// const API_BASE_URL = 'http://your-backend-domain.com';
// const API_BASE_URL = 'https://your-backend-domain.com';
// const API_BASE_URL = 'http://192.168.1.100:5000';

// API Endpoints
const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/api/v1/health`,
  USERS: `${API_BASE_URL}/api/v1/users`,
  REGISTER: `${API_BASE_URL}/api/v1/register`,
  VERIFY: `${API_BASE_URL}/api/v1/verify`,
  CHECK_DUPLICATE: `${API_BASE_URL}/api/v1/check-duplicate`,
  VERIFICATION_LOGS: `${API_BASE_URL}/api/v1/verification-logs`,
  ENCODINGS: `${API_BASE_URL}/api/v1/encodings`,
  
  // Legacy Flask routes
  VIDEO_FEED: `${API_BASE_URL}/video_feed`,
  VERIFICATION_STATUS: `${API_BASE_URL}/verification_status`,
  STOP_VERIFICATION: `${API_BASE_URL}/stop_verification`,
  UPLOAD: `${API_BASE_URL}/upload`,
  CHECK_DUPLICATE_FORM: `${API_BASE_URL}/check_duplicate`,
  INDEX: `${API_BASE_URL}/`,
  CAMERA: `${API_BASE_URL}/camera`,
  REGISTER_PAGE: `${API_BASE_URL}/register`,
};

console.log(`🔗 Frontend configured to use backend at: ${API_BASE_URL}`);
