"""
Database Initialization Script
Run this to create database tables on first setup
"""

import os
import sys
from app import app, db
from models import User, FaceEncoding, VerificationLog


def init_database():
    """Initialize database tables"""
    with app.app_context():
        try:
            print("Initializing database...")
            db.create_all()
            print("✓ Database tables created successfully!")
            
            # Verify tables exist
            inspector_result = db.inspect(db.engine)
            tables = inspector_result.get_table_names()
            
            print(f"\nCreated tables:")
            for table in tables:
                if table in ['users', 'face_encodings', 'verification_logs']:
                    print(f"  ✓ {table}")
            
            print("\n✓ Database initialization complete!")
            return True
            
        except Exception as e:
            print(f"✗ Error during database initialization: {str(e)}")
            return False


def verify_connection():
    """Verify database connection"""
    with app.app_context():
        try:
            result = db.session.execute(db.text("SELECT 1"))
            print("✓ Database connection successful!")
            return True
        except Exception as e:
            print(f"✗ Database connection failed: {str(e)}")
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("Face Recognition Database Initializer")
    print("=" * 60)
    
    # Check environment
    print("\nChecking configuration...")
    from config import Config
    print(f"Database: {Config.MYSQL_DATABASE}")
    print(f"Host: {Config.MYSQL_HOST}:{Config.MYSQL_PORT}")
    print(f"User: {Config.MYSQL_USER}")
    
    # Verify connection
    print("\nVerifying database connection...")
    if not verify_connection():
        print("\n⚠ Connection failed. Please check:")
        print("  - MySQL server is running")
        print("  - Database exists")
        print("  - Credentials in .env are correct")
        sys.exit(1)
    
    # Initialize database
    print("\nInitializing database tables...")
    if init_database():
        print("\n" + "=" * 60)
        print("Setup complete! You can now start the application:")
        print("  python app.py")
        print("=" * 60)
    else:
        sys.exit(1)
