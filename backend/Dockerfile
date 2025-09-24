# Stage 1: Use an official Python slim image as a base
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Install system dependencies required by OpenCV
# This is crucial for image processing to work correctly
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's caching mechanism
# If requirements don't change, this layer won't be rebuilt, speeding up future builds.
COPY ./requirements.txt /app/requirements.txt

# Install the Python dependencies.
# Make sure your requirements.txt lists `onnxruntime` (or `-gpu`) and `scipy`
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the entire backend application code into the container's /app directory
COPY . /app

# Expose the port that the Uvicorn server will run on
EXPOSE 8000

# The command to run when the container starts
# --host 0.0.0.0 is essential to make the server accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]