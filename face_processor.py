import face_recognition
import os
import numpy as np
import cv2
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from .face_vector import FaceEmbeddingDB

class FaceProcessor:
    def __init__(self, employee_images_path: str, db_handler: FaceEmbeddingDB):
        self.employee_images_path = employee_images_path
        self.db_handler = db_handler
        self.last_reload_time = None
        self.reload_interval = timedelta(minutes=10)
        self.process_employee_images()

    def should_reload(self) -> bool:
        """Check if we should reload the encodings based on time interval."""
        if self.last_reload_time is None:
            return True
        return datetime.now() - self.last_reload_time > self.reload_interval

    def process_employee_images(self):
        """Process and store employee image encodings in the database."""
        print("Processing employee images...")
        
        for employee_name in os.listdir(self.employee_images_path):
            employee_dir = os.path.join(self.employee_images_path, employee_name)
            if os.path.isdir(employee_dir):
                for image_file in os.listdir(employee_dir):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(employee_dir, image_file)
                        self._process_employee_image(image_path, employee_name)
        
        self.last_reload_time = datetime.now()
        print(f"Finished processing employee images")

    def _process_employee_image(self, image_path: str, employee_name: str):
        """Process a single employee image and return its encoding."""
        try:
            # Check if the embedding already exists in the database
            if self.db_handler.embedding_exists(employee_name):
                print(f"Embedding for {employee_name} already exists in the database.")
                return None
            
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                encoding = face_encodings[0]
                self.db_handler.store_embedding(employee_name, encoding)
                return encoding
            return None
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def process_video_frame(self, frame_numpy: np.ndarray) -> List[Dict]:
        """Process a single video frame and return detected faces with enhanced matching logic."""
        # Check if we need to reload employee images
        if self.should_reload():
            self.process_employee_images()
            
        # Reduce frame size minimally for better accuracy
        small_frame = cv2.resize(frame_numpy, (0, 0), fx=0.35, fy=0.35)
        
        # Convert from BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        detected_faces = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Get top 5 closest matches from vector search
            results = self.db_handler.vector_search(face_encoding)
            
            name = "Unknown"
            confidence = 0.0
            
            if results:
                try:
                    candidate_encodings = []
                    for result in results:
                        embedding_str = result["embedding"].strip('[]')
                        embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
                        candidate_encodings.append(np.array(embedding_values))
                    
                    candidate_names = [result["name"] for result in results]
                    
                    # Compare with candidate faces using face_recognition
                    matches = face_recognition.compare_faces(candidate_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(candidate_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = candidate_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                            # Also consider the vector similarity as a factor
                            vector_confidence = results[best_match_index]["similarity"]
                            # Take the average of both confidence measures
                            confidence = (confidence + vector_confidence) / 2
                except Exception as e:
                    print(f"Error processing embeddings: {e}")
                    continue
            
            # Only include if confidence is high enough
            if name == "Unknown" or confidence > 0.7:
                top, right, bottom, left = face_location
                detected_faces.append({
                    "name": name,
                    "confidence": float(confidence),
                    "location": {
                        "top": top * 4,
                        "right": right * 4,
                        "bottom": bottom * 4,
                        "left": left * 4
                    }
                })
        
        return detected_faces