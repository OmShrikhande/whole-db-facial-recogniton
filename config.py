import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'root')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'face_recognition_db')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    
    password_part = f":{MYSQL_PASSWORD}" if MYSQL_PASSWORD else ""
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{MYSQL_USER}{password_part}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': int(os.getenv('DB_POOL_SIZE', 10)),
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 20)),
    }
    
    FLASK_SECRET = os.getenv('FLASK_SECRET', 'change-me-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5000')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5000').split(',')
    
    SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', 3600))
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 25))
    DUPLICATE_THRESHOLD = float(os.getenv('DUPLICATE_THRESHOLD', 0.45))
    VERIFICATION_THRESHOLD = float(os.getenv('VERIFICATION_THRESHOLD', 0.45))
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'simple')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', 300))
    
    BACKUP_DIR = os.getenv('BACKUP_DIR', '/backups/face-recognition')
    BACKUP_RETENTION_DAYS = int(os.getenv('BACKUP_RETENTION_DAYS', 30))
