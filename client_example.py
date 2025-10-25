import requests
import json
from pathlib import Path

class FaceRecognitionClient:
    def __init__(self, base_url="http://localhost:5000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check system health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_all_users(self):
        """Get all registered users"""
        response = self.session.get(f"{self.base_url}/users")
        return response.json()
    
    def get_user(self, name):
        """Get specific user by name"""
        response = self.session.get(f"{self.base_url}/users/{name}")
        return response.json()
    
    def register_user(self, name, image_path):
        """
        Register a new user with face encoding
        
        Args:
            name (str): User's name
            image_path (str): Path to face image
        
        Returns:
            dict: Response data
        """
        with open(image_path, 'rb') as img_file:
            files = {'file': img_file}
            data = {'name': name}
            response = self.session.post(f"{self.base_url}/register", files=files, data=data)
        return response.json()
    
    def verify_face(self, image_path):
        """
        Verify a face image against registered users
        
        Args:
            image_path (str): Path to face image
        
        Returns:
            dict: Verification result
        """
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            response = self.session.post(f"{self.base_url}/verify", files=files)
        return response.json()
    
    def check_duplicate(self, image_path):
        """
        Check if face is duplicate of existing users
        
        Args:
            image_path (str): Path to face image
        
        Returns:
            dict: Duplicate check result
        """
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            response = self.session.post(f"{self.base_url}/check-duplicate", files=files)
        return response.json()
    
    def get_verification_logs(self, limit=100):
        """Get verification attempt logs"""
        response = self.session.get(
            f"{self.base_url}/verification-logs",
            params={'limit': limit}
        )
        return response.json()
    
    def get_encodings(self):
        """Get all face encodings for client-side processing"""
        response = self.session.get(f"{self.base_url}/encodings")
        return response.json()


def main():
    client = FaceRecognitionClient()
    
    print("=" * 60)
    print("Face Recognition System - Client Example")
    print("=" * 60)
    
    try:
        # Health check
        print("\n1. Health Check:")
        health = client.health_check()
        print(json.dumps(health, indent=2))
        
        # Get all users
        print("\n2. Get All Users:")
        users = client.get_all_users()
        print(json.dumps(users, indent=2))
        
        # Verify logs
        print("\n3. Get Verification Logs (limit 10):")
        logs = client.get_verification_logs(limit=10)
        print(json.dumps(logs, indent=2))
        
        # Register user example (requires actual image file)
        print("\n4. Register User Example:")
        print("   Syntax: client.register_user('Name', '/path/to/image.jpg')")
        print("   Returns: User data with ID")
        
        # Verify example
        print("\n5. Verify Face Example:")
        print("   Syntax: client.verify_face('/path/to/image.jpg')")
        print("   Returns: Verification status and matched user")
        
        # Check duplicate example
        print("\n6. Check Duplicate Example:")
        print("   Syntax: client.check_duplicate('/path/to/image.jpg')")
        print("   Returns: Duplicate status and matches")
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API server")
        print("Make sure the Flask app is running on http://localhost:5000")
    except Exception as e:
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main()
