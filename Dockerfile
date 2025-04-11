FROM python:3.9.18-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app ./app

# Command to run the application with reload enabled
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
