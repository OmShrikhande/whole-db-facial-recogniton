# Face Recognition System - Professional Setup Guide

## Prerequisites
- Python 3.7+
- MySQL Server 5.7+
- Git (optional)

---

## Installation Steps

### 1. Clone/Extract Project
```bash
cd your-project-directory
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Database

**Create MySQL Database:**
```bash
mysql -u root -p
```

```sql
CREATE DATABASE face_recognition_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'fr_user'@'localhost' IDENTIFIED BY 'secure_password';
GRANT ALL PRIVILEGES ON face_recognition_db.* TO 'fr_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 5. Configure Environment Variables

Create `.env` file in project root:
```env
MYSQL_HOST=localhost
MYSQL_USER=fr_user
MYSQL_PASSWORD=secure_password
MYSQL_DATABASE=face_recognition_db
MYSQL_PORT=3306
FLASK_SECRET=generate-secure-random-key-here
DEBUG=False
```

**Generate Secret Key (Python):**
```python
import secrets
print(secrets.token_hex(32))
```

### 6. Initialize Database
```bash
python
```

```python
from app import app, db
with app.app_context():
    db.create_all()
    print("Database tables created!")
exit()
```

### 7. Run Application

**Development:**
```bash
python app.py
```

**Production (using Gunicorn):**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Access at: `http://localhost:5000`

---

## Production Deployment

### Using Apache + mod_wsgi

**1. Install mod_wsgi:**
```bash
pip install mod_wsgi
```

**2. Create `app.wsgi`:**
```python
import sys
sys.path.insert(0, '/path/to/project')

from app import app as application
```

**3. Apache VirtualHost:**
```apache
<VirtualHost *:80>
    ServerName yourdomain.com
    DocumentRoot /path/to/project

    WSGIDaemonProcess face_rec user=www-data group=www-data threads=5
    WSGIScriptAlias / /path/to/project/app.wsgi

    <Directory /path/to/project>
        WSGIProcessGroup face_rec
        WSGIApplicationGroup %{GLOBAL}
        Require all granted
    </Directory>

    ErrorLog ${APACHE_LOG_DIR}/face_rec_error.log
    CustomLog ${APACHE_LOG_DIR}/face_rec_access.log combined
</VirtualHost>
```

### Using Nginx + Gunicorn

**1. Install Supervisor:**
```bash
apt-get install supervisor
```

**2. Create `/etc/supervisor/conf.d/face_rec.conf`:**
```ini
[program:face_rec]
command=/path/to/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
directory=/path/to/project
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/face_rec/err.log
stdout_logfile=/var/log/face_rec/out.log
```

**3. Nginx Configuration:**
```nginx
upstream face_rec {
    server 127.0.0.1:5000;
}

server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://face_rec;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static {
        alias /path/to/project/static;
    }
}
```

---

## Database Backup & Recovery

### Backup
```bash
mysqldump -u root -p face_recognition_db > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Restore
```bash
mysql -u root -p face_recognition_db < backup_file.sql
```

---

## Monitoring & Logging

### Application Logs
Logs are printed to console. For production, redirect to file:
```bash
python app.py > logs/app.log 2>&1 &
```

### Database Connection Issues
Check MySQL service:
```bash
# Windows
net start MySQL80

# Linux
sudo systemctl start mysql
```

---

## Security Checklist

- [ ] Change `FLASK_SECRET` to a strong random key
- [ ] Use strong MySQL passwords
- [ ] Set `DEBUG=False` in production
- [ ] Use HTTPS/SSL certificates
- [ ] Implement API authentication (JWT/API Keys)
- [ ] Set up firewall rules
- [ ] Regular database backups
- [ ] Restrict file upload sizes
- [ ] Sanitize user inputs
- [ ] Monitor API access logs

---

## API Base URL

- **Development:** `http://localhost:5000/api/v1`
- **Production:** `https://yourdomain.com/api/v1`

---

## Troubleshooting

### Database Connection Error
```
Error: Can't connect to MySQL server on 'localhost'
```
**Solution:** Ensure MySQL is running and credentials are correct in `.env`

### Face Recognition Library Issues
```
Error: face_recognition module not found
```
**Solution:** Install dependencies with specific versions:
```bash
pip install dlib==19.24.1
pip install face_recognition==1.3.5
```

### Permission Denied for known_faces/unknown_faces
**Solution:**
```bash
chmod 755 known_faces unknown_faces
```

---

## API Testing

Use provided Postman collection or test with cURL:

```bash
# Health check
curl http://localhost:5000/api/v1/health

# Get users
curl http://localhost:5000/api/v1/users

# Register user
curl -X POST http://localhost:5000/api/v1/register \
  -F "name=Test User" \
  -F "file=@photo.jpg"
```

---

## Support
For issues, refer to `API_DOCUMENTATION.md` or contact support.
