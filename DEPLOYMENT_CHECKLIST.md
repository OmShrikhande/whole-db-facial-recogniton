# Face Recognition System - Deployment Checklist

## Pre-Deployment

### Environment Setup
- [ ] Python 3.7+ installed
- [ ] MySQL Server 5.7+ installed and running
- [ ] Git installed (optional, for version control)
- [ ] Virtual environment created
  ```bash
  python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  ```

### Dependencies
- [ ] Dependencies installed
  ```bash
  pip install -r requirements.txt
  ```
- [ ] All packages verified
  ```bash
  pip list | grep -E "Flask|SQLAlchemy|face-recognition|opencv|numpy"
  ```

---

## Database Setup

### MySQL Database
- [ ] MySQL Server running
  - **Windows**: `net start MySQL80` or Services
  - **Linux**: `sudo systemctl start mysql`
  - **macOS**: `brew services start mysql`

- [ ] MySQL root user accessible
  ```bash
  mysql -u root -p
  ```

- [ ] Database created
  ```bash
  mysql -u root -p -e "CREATE DATABASE face_recognition_db CHARACTER SET utf8mb4;"
  ```

- [ ] Database user created (recommended)
  ```sql
  CREATE USER 'fr_user'@'localhost' IDENTIFIED BY 'secure_password';
  GRANT ALL PRIVILEGES ON face_recognition_db.* TO 'fr_user'@'localhost';
  FLUSH PRIVILEGES;
  ```

---

## Configuration

### Environment Variables
- [ ] `.env` file created in project root
- [ ] `.env` contains:
  ```env
  MYSQL_HOST=localhost
  MYSQL_USER=root
  MYSQL_PASSWORD=your_password
  MYSQL_DATABASE=face_recognition_db
  MYSQL_PORT=3306
  FLASK_SECRET=your-secure-random-key
  DEBUG=False
  ```
- [ ] Sensitive credentials NOT committed to git
- [ ] `.gitignore` includes `.env`

### Flask Secret Key
- [ ] Secret key generated and set
  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```
- [ ] Secret key added to `.env`

---

## Database Initialization

### Tables Creation
- [ ] Database tables initialized
  ```bash
  python init_db.py
  ```
- [ ] Output shows successful table creation
- [ ] Tables verified in MySQL
  ```sql
  USE face_recognition_db;
  SHOW TABLES;
  ```

### Verify Schema
- [ ] Users table exists with correct columns
- [ ] Face encodings table exists with correct columns
- [ ] Verification logs table exists with correct columns
- [ ] Foreign key relationships verified

---

## Application Testing

### Development Mode
- [ ] Application starts without errors
  ```bash
  python app.py
  ```
- [ ] Application accessible at `http://localhost:5000`
- [ ] Web interface loads correctly
- [ ] No Python errors in console

### API Health Check
- [ ] Health endpoint responds
  ```bash
  curl http://localhost:5000/api/v1/health
  ```
- [ ] Returns 200 status with healthy response
- [ ] Shows database connection status

### Database Connectivity
- [ ] Test registration via API
  ```bash
  curl -X POST http://localhost:5000/api/v1/register \
    -F "name=Test User" \
    -F "file=@test_image.jpg"
  ```
- [ ] User appears in database
  ```bash
  curl http://localhost:5000/api/v1/users
  ```

### Face Recognition
- [ ] Face verification works
  ```bash
  curl -X POST http://localhost:5000/api/v1/verify \
    -F "image=@test_image.jpg"
  ```
- [ ] Returns correct verification status

---

## Security Hardening

### File Permissions
- [ ] Project directory permissions set correctly
  ```bash
  chmod 755 known_faces unknown_faces templates
  chmod 600 .env
  ```
- [ ] .env file is not readable by others

### Secrets Management
- [ ] All secrets in `.env`, not in code
- [ ] `.env` added to `.gitignore`
- [ ] No passwords in source files
- [ ] No API keys in templates

### Input Validation
- [ ] File upload size limits set
- [ ] Only image files accepted
- [ ] User input sanitized
- [ ] Database queries use ORM (SQL injection prevention)

---

## Production Deployment

### Choose Deployment Option

#### Option A: Gunicorn
- [ ] Gunicorn installed
  ```bash
  pip install gunicorn
  ```
- [ ] Test with Gunicorn
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5000 app:app
  ```
- [ ] Runs without errors
- [ ] Accessible from external IP

#### Option B: Apache + mod_wsgi
- [ ] Apache installed and running
- [ ] mod_wsgi installed
  ```bash
  pip install mod_wsgi
  ```
- [ ] Apache configuration created
- [ ] Virtual host configured
- [ ] Application accessible via Apache

#### Option C: Nginx + Supervisor
- [ ] Nginx installed and running
- [ ] Supervisor installed
  ```bash
  sudo apt-get install supervisor
  ```
- [ ] Gunicorn running under Supervisor
- [ ] Nginx proxy configured
- [ ] Application accessible via Nginx

### SSL/HTTPS Setup (Recommended)
- [ ] SSL certificate obtained (Let's Encrypt or commercial)
- [ ] Certificate configured in web server
- [ ] HTTPS enforced (redirect HTTP to HTTPS)
- [ ] Certificate renewal scheduled

### Performance Tuning
- [ ] Gunicorn workers optimized
  ```bash
  # Formula: (2 × CPU cores) + 1
  gunicorn -w 9 -b 0.0.0.0:5000 app:app
  ```
- [ ] Database connection pooling configured
- [ ] Caching headers set appropriately
- [ ] Compression enabled

---

## Monitoring Setup

### Logging
- [ ] Application logs directed to file
  ```bash
  python app.py >> logs/app.log 2>&1 &
  ```
- [ ] Log rotation configured
- [ ] Log directory created with proper permissions

### Health Monitoring
- [ ] Health check endpoint configured
  ```bash
  curl http://your-domain.com/api/v1/health
  ```
- [ ] Automated health check scheduled (cron)
- [ ] Alerts set up for failures

### Database Monitoring
- [ ] Database connection monitoring
- [ ] Disk space monitoring
- [ ] Query performance monitoring
- [ ] Backup verification

---

## Backup & Recovery

### Backup Strategy
- [ ] Database backup script created
  ```bash
  #!/bin/bash
  mysqldump -u fr_user -p face_recognition_db > /backup/db_$(date +%Y%m%d).sql
  ```
- [ ] Backup scheduled (cron job)
- [ ] Backup location verified
- [ ] Multiple backups retained
- [ ] Backup storage on separate drive/server

### Disaster Recovery
- [ ] Recovery procedure documented
- [ ] Test restore from backup completed successfully
- [ ] Recovery time objective (RTO) defined
- [ ] Recovery point objective (RPO) defined

### Data Retention
- [ ] Old logs cleanup policy defined
  ```sql
  DELETE FROM verification_logs 
  WHERE created_at < DATE_SUB(NOW(), INTERVAL 90 DAY);
  ```
- [ ] Cleanup scheduled as cron job

---

## Documentation

### API Documentation
- [ ] `API_DOCUMENTATION.md` reviewed
- [ ] All endpoints documented
- [ ] Client examples tested
- [ ] Error responses documented

### Setup Documentation
- [ ] `SETUP_GUIDE.md` reviewed
- [ ] Configuration steps verified
- [ ] Deployment instructions clear
- [ ] Troubleshooting guide complete

### Operational Documentation
- [ ] Standard operating procedures documented
- [ ] Common issues and solutions documented
- [ ] Contact information for support
- [ ] Escalation procedures defined

---

## Client Handoff

### Code Delivery
- [ ] Source code organized and clean
- [ ] Comments added where necessary
- [ ] README files complete
- [ ] All documentation included

### Client Training
- [ ] Client trained on API usage
- [ ] Client trained on system monitoring
- [ ] Client trained on backup/recovery
- [ ] Client given contact information

### Support Plan
- [ ] Support contact information provided
- [ ] Support hours defined
- [ ] SLA defined (if applicable)
- [ ] Issue reporting procedure documented

---

## Post-Deployment

### Monitoring (First Week)
- [ ] Application running without errors
- [ ] Database performing well
- [ ] No memory leaks detected
- [ ] Error logs reviewed daily
- [ ] Performance baseline established

### Optimization
- [ ] Performance monitoring enabled
- [ ] Database queries optimized if needed
- [ ] Caching implemented if needed
- [ ] Load testing completed

### Documentation Updates
- [ ] Operations manual updated
- [ ] Known issues documented
- [ ] Performance metrics recorded
- [ ] Lessons learned captured

---

## Compliance & Security

### Security Audit
- [ ] HTTPS/SSL enabled
- [ ] API authentication implemented
- [ ] Input validation verified
- [ ] SQL injection prevention verified
- [ ] CORS policies set appropriately

### Compliance
- [ ] Data privacy policy reviewed
- [ ] GDPR compliance (if applicable)
- [ ] Data retention policies implemented
- [ ] User consent management

### Access Control
- [ ] Admin access restricted
- [ ] SSH key-based authentication
- [ ] Database user has minimal privileges
- [ ] Firewall rules configured

---

## Troubleshooting Checklist

### If application won't start:
- [ ] Check MySQL is running: `mysql -u root -p`
- [ ] Check .env file exists with correct credentials
- [ ] Check database exists: `SHOW DATABASES;`
- [ ] Check tables exist: `SHOW TABLES;`
- [ ] Check Python errors: `python app.py`

### If API returns 500 errors:
- [ ] Check application logs
- [ ] Verify database connection
- [ ] Check database has data
- [ ] Verify API endpoint syntax

### If face recognition not working:
- [ ] Verify image file is valid
- [ ] Check image size (minimum 80x80 pixels)
- [ ] Verify face is clearly visible
- [ ] Check multiple faces not detected

### If database errors occur:
- [ ] Check MySQL is running and accessible
- [ ] Verify credentials in .env
- [ ] Check disk space on database server
- [ ] Review MySQL error logs

---

## Completion Checklist

- [ ] All items above completed
- [ ] System tested in production environment
- [ ] Client trained and comfortable with system
- [ ] Documentation complete and delivered
- [ ] Support plan in place
- [ ] Monitoring and alerts configured
- [ ] Backup and recovery procedures tested
- [ ] Go-live approval obtained

---

## Sign-Off

**System Administrator Name:** ___________________________  
**Date:** ___________________________  
**Sign-Off:** ✓  

**Client Representative:** ___________________________  
**Date:** ___________________________  
**Sign-Off:** ✓  

---

## Notes

```
[Space for deployment notes and issues encountered]
```

---

**Status**: Ready for Deployment ✅
