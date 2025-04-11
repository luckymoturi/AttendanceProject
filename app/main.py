import psycopg2
import numpy as np
from typing import List, Dict, Optional
from psycopg2.extras import execute_values
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .face_processor import FaceProcessor
from .face_vector import FaceEmbeddingDB
import cv2
import tempfile
import os
import face_recognition
import math

# Example geofence locations (latitude, longitude)
GEOFENCES = {
    "office": (16.5422428,81.4968464),  # Example coordinates for Bangalore
}

# Maximum allowed distance in meters to be considered within the geofence
MAX_DISTANCE = 100

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

db_params = {
    'dbname': 'face_recognition',
    'user': 'postgres',
    'password': 'password',
    'host': 'pgvector-db',
    'port': '5432'
}

# Initialize database handler
db_handler = FaceEmbeddingDB(db_params)

# Initialize face processor
face_processor = FaceProcessor("./employee_images", db_handler)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points."""
    R = 6371  # Radius of the Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c * 1000  # Convert to meters
    return distance

@app.post("/enroll-photo/")
async def enroll_photo(name: str, photo: UploadFile = File(...)):
    """
    Upload a photo to generate embeddings and store them in the vector database.

    Args:
        name: The name of the person to enroll (as a query parameter).
        photo: The photo file to upload.

    Returns:
        A message indicating success or failure.
    """
    try:
        # Input validation
        if not name or name.strip() == "":
            return {"status_code": 400, "message": "Name cannot be empty"}

        # Clean the name (remove leading/trailing spaces)
        name = name.strip()

        # Define the directory path
        employee_images_dir = r"Attendance\my-app\src\Backend\Face_attendance\employee_images"
        employee_dir = os.path.join(employee_images_dir, name)

        # Create the directory if it doesn't exist
        os.makedirs(employee_dir, exist_ok=True)

        # Define the path for the photo
        photo_path = os.path.join(employee_dir, f"{name}.jpg")

        # Save the uploaded photo to the specified path
        with open(photo_path, "wb") as f:
            content = await photo.read()
            f.write(content)

        # Generate embedding
        encoding = face_processor._process_employee_image(photo_path, name)
        if encoding is None:
            return {"status_code": 400, "message": "Failed to generate embedding."}

        # Check if a similar embedding already exists
        similar_embeddings = db_handler.vector_search(encoding)
        if similar_embeddings:
            return {"status_code": 400, "message": f"User with similar face already exists."}

        # Store the embedding in the database
        if db_handler.store_embedding(name, encoding):
            return {"message": f"Embedding stored successfully for {name}."}
        else:
            return {"status_code": 400, "message": "Failed to store embedding."}

    except Exception as e:
        return {"status_code": 500, "message": f"Error: {str(e)}"}



@app.post("/process-video")
async def process_video(video: UploadFile = File(...)):
    """Process uploaded video and detect faces."""
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        content = await video.read()
        temp_video.write(content)
        temp_video_path = temp_video.name

    try:
        results = []
        cap = cv2.VideoCapture(temp_video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 5th frame to improve performance
            if frame_count % 5 == 0:
                detected_faces = face_processor.process_video_frame(frame)
                if detected_faces:
                    results.append({
                        "frame": frame_count,
                        "detected_faces": detected_faces
                    })

            frame_count += 1

        cap.release()

        return {
            "total_frames": frame_count,
            "processed_frames": len(results),
            "detections": results
        }

    finally:
        # Cleanup temporary file
        os.unlink(temp_video_path)


@app.post("/process-checkin")
async def process_checkin(photo: UploadFile = File(...), latitude: float = 16.5422428, longitude: float = 81.4968464):
    """Process check-in photo and detect faces with geofencing."""
    try:
        # Check if the user is within any geofence
        user_location = (latitude, longitude)
        is_within_geofence = any(
            haversine(user_location[0], user_location[1], geofence[0], geofence[1]) <= MAX_DISTANCE
            for geofence in GEOFENCES.values()
        )

        if not is_within_geofence:
            return {"success": False, "message": "Check-in failed. You are not within the allowed area."}

        # Save the uploaded photo to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_photo:
            content = await photo.read()
            temp_photo.write(content)
            temp_photo_path = temp_photo.name

        # Process the photo to detect faces
        detected_faces = face_processor.process_video_frame(cv2.imread(temp_photo_path))

        # Cleanup temporary file
        os.unlink(temp_photo_path)

        if detected_faces and detected_faces[0]['name'] != 'Unknown':
            user_name = detected_faces[0]['name']

            # Check if the user has already checked in today
            if db_handler.has_checked_in_today(user_name):
                return {"success": False, "message": "You have already checked in today."}

            db_handler.log_attendance(user_name, 'checkin', latitude, longitude)
            return {"success": True, "message": "Check-in successful!"}
        else:
            return {"success": False, "message": "Face not recognized."}

    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/process-checkout")
async def process_checkout(photo: UploadFile = File(...), latitude: float = 16.5422428, longitude: float = 81.4968464):
    """Process check-out photo and detect faces with geofencing."""
    try:
        # Check if the user is within any geofence
        user_location = (latitude, longitude)
        is_within_geofence = any(
            haversine(user_location[0], user_location[1], geofence[0], geofence[1]) <= MAX_DISTANCE
            for geofence in GEOFENCES.values()
        )

        if not is_within_geofence:
            return {"success": False, "message": "Check-out failed. You are not within the allowed area."}

        # Save the uploaded photo to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_photo:
            content = await photo.read()
            temp_photo.write(content)
            temp_photo_path = temp_photo.name

        # Process the photo to detect faces
        detected_faces = face_processor.process_video_frame(cv2.imread(temp_photo_path))

        # Cleanup temporary file
        os.unlink(temp_photo_path)

        if detected_faces and detected_faces[0]['name'] != 'Unknown':
            user_name = detected_faces[0]['name']

            # Check if the user has already checked out today
            if db_handler.has_checked_out_today(user_name):
                return {"success": False, "message": "You have already checked out today."}

            db_handler.log_attendance(user_name, 'checkout', latitude, longitude)
            return {"success": True, "message": "Check-out successful!"}
        else:
            return {"success": False, "message": "Face not recognized."}

    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.get("/getall")
async def find():
    results = db_handler.retrieve_all_data()
    return results

@app.get("/delete")
async def delete():
    db_handler.delete_tables()
    return {"message": "Tables deleted"}

@app.delete("/delete-user/{name}")
async def delete_user(name: str):
    try:
        success = db_handler.delete_user(name)
        if success:
            return {"status": "success", "message": f"User '{name}' deleted successfully"}
        else:
            return {"status": "error", "message": f"User '{name}' not found"}
    except Exception as e:
        return {"status": "error", "message": f"Error deleting user: {str(e)}"}

@app.get("/attendance/{user_name}")
async def get_attendance(user_name: str):
    """Fetch attendance records for a specific user."""
    attendance_records = db_handler.retrieve_attendance(user_name)
    return attendance_records

@app.get("/user-report/{user_name}")
async def get_user_report(user_name: str):
    """
    Get detailed attendance report for a specific user.

    Args:
        user_name: The name of the user to get the report for

    Returns:
        List of daily attendance records with check-in and check-out times
    """
    try:
        report = db_handler.get_user_attendance_report(user_name)
        if report:
            return {
                "status": "success",
                "user_name": user_name,
                "attendance_records": report
            }
        else:
            return {
                "status": "error",
                "message": f"No attendance records found for user '{user_name}'"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving attendance report: {str(e)}"
        }


@app.get("/notfound")
async def notfound():
    return {"message": "Table not found"}
