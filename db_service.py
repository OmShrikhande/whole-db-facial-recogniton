from models import db, User, FaceEncoding, VerificationLog
import face_recognition
import numpy as np
from datetime import datetime

class DatabaseService:
    
    @staticmethod
    def register_user(name, image_data, image_path=None):
        try:
            existing_user = User.query.filter_by(name=name).first()
            if existing_user:
                return False, "User already exists"
            
            image = face_recognition.load_image_file(image_data)
            encs = face_recognition.face_encodings(image)
            
            if len(encs) == 0:
                return False, "No face found in image"
            if len(encs) > 1:
                return False, "Multiple faces detected"
            
            user = User(name=name)
            db.session.add(user)
            db.session.flush()
            
            encoding_obj = FaceEncoding(user_id=user.id, image_path=image_path)
            encoding_obj.set_encoding(encs[0])
            db.session.add(encoding_obj)
            db.session.commit()
            
            return True, f"User {name} registered successfully"
        except Exception as e:
            db.session.rollback()
            return False, str(e)
    
    @staticmethod
    def check_duplicate(image_data):
        try:
            image = face_recognition.load_image_file(image_data)
            encs = face_recognition.face_encodings(image)
            
            if len(encs) == 0:
                return {"status": "no_face"}
            if len(encs) > 1:
                return {"status": "multiple_faces"}
            
            encoding = encs[0]
            active_encodings = FaceEncoding.query.filter_by(is_active=True).all()
            
            if not active_encodings:
                return {"duplicate": False}
            
            threshold = 0.45
            matches = []
            best_distance = 1.0
            
            for enc_obj in active_encodings:
                enc_array = enc_obj.get_encoding()
                distance = face_recognition.face_distance([enc_array], encoding)[0]
                
                if distance < best_distance:
                    best_distance = float(distance)
                
                if distance <= threshold:
                    user = User.query.get(enc_obj.user_id)
                    if user and user not in matches:
                        matches.append(user.name)
            
            if matches:
                return {"duplicate": True, "matches": matches, "best_distance": best_distance}
            return {"duplicate": False, "best_distance": best_distance}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def verify_face(image_data):
        try:
            image = face_recognition.load_image_file(image_data)
            faces = face_recognition.face_locations(image)
            
            if len(faces) == 0:
                return {"status": "no_face", "message": "No face detected"}
            if len(faces) > 1:
                return {"status": "multiple_faces", "message": "Multiple faces detected"}
            
            encs = face_recognition.face_encodings(image, faces)
            if len(encs) == 0:
                return {"status": "unknown", "message": "Could not encode face"}
            
            test_encoding = encs[0]
            active_encodings = FaceEncoding.query.filter_by(is_active=True).all()
            
            if not active_encodings:
                return {"status": "unknown", "message": "No registered faces"}
            
            threshold = 0.45
            matches = {}
            
            for enc_obj in active_encodings:
                enc_array = enc_obj.get_encoding()
                distance = face_recognition.face_distance([enc_array], test_encoding)[0]
                
                if distance <= threshold:
                    user = User.query.get(enc_obj.user_id)
                    if user:
                        if user.id not in matches:
                            matches[user.id] = {"name": user.name, "distance": float(distance)}
                        else:
                            if distance < matches[user.id]["distance"]:
                                matches[user.id]["distance"] = float(distance)
            
            if matches:
                best_match = min(matches.items(), key=lambda x: x[1]["distance"])
                user_id = best_match[0]
                user_name = best_match[1]["name"]
                
                log = VerificationLog(user_id=user_id, status="verified", message=f"Verified as {user_name}")
                db.session.add(log)
                db.session.commit()
                
                return {"status": "verified", "message": f"Verified as {user_name}", "user_id": user_id, "name": user_name}
            else:
                log = VerificationLog(status="unknown", message="Unknown face")
                db.session.add(log)
                db.session.commit()
                return {"status": "unknown", "message": "Unknown face"}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    def get_all_users():
        users = User.query.filter_by(is_active=True).all()
        return [user.to_dict() for user in users]
    
    @staticmethod
    def get_user_by_name(name):
        user = User.query.filter_by(name=name, is_active=True).first()
        return user.to_dict() if user else None
    
    @staticmethod
    def get_encodings_for_verification():
        encodings = []
        names = []
        
        active_encodings = FaceEncoding.query.filter_by(is_active=True).all()
        for enc_obj in active_encodings:
            encodings.append(enc_obj.get_encoding())
            user = User.query.get(enc_obj.user_id)
            if user:
                names.append(user.name)
        
        return encodings, names
    
    @staticmethod
    def add_encoding_for_user(user_id, image_data, image_path=None):
        try:
            image = face_recognition.load_image_file(image_data)
            encs = face_recognition.face_encodings(image)
            
            if len(encs) == 0:
                return False, "No face found"
            if len(encs) > 1:
                return False, "Multiple faces detected"
            
            encoding_obj = FaceEncoding(user_id=user_id, image_path=image_path)
            encoding_obj.set_encoding(encs[0])
            db.session.add(encoding_obj)
            db.session.commit()
            
            return True, "Encoding added"
        except Exception as e:
            db.session.rollback()
            return False, str(e)
    
    @staticmethod
    def log_verification(user_id, status, message, image_path=None):
        log = VerificationLog(user_id=user_id, status=status, message=message, image_path=image_path)
        db.session.add(log)
        db.session.commit()
        return log.to_dict()
    
    @staticmethod
    def get_verification_logs(limit=100):
        logs = VerificationLog.query.order_by(VerificationLog.created_at.desc()).limit(limit).all()
        return [log.to_dict() for log in logs]
