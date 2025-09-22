from fastapi import APIRouter, Depends, Request, UploadFile, File, Body
import shutil
import os
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.services import database_service as db_service
from app.db.session import get_db
# It's a good practice to get settings from your config file
from app.core.config import settings
from ...db import models
from datetime import datetime, timedelta

router = APIRouter()

# BEST PRACTICE: Move the video path to your settings/config file.
# For now, we'll use the one from your code.
VIDEO_PATH = r"C:\Users\momen\Downloads\2103099-uhd_3840_2160_30fps.mp4"
# Or, if you add it to your config: VIDEO_PATH = settings.VIDEO_FILE_PATH

# Create a directory to store uploads
UPLOADS_DIR = "uploaded_videos"
os.makedirs(UPLOADS_DIR, exist_ok=True)

@router.get("/video-feed")
def video_feed(request: Request):
    """
    Streams video from the currently configured source (file or camera).
    """
    processor = request.app.state.traffic_processor
    source_config = request.app.state.video_source
    
    # Determine the video path or device ID from the state
    video_path_or_id = source_config["path"]

    return StreamingResponse(
        processor.process_video(video_path_or_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/latest-metrics")
def get_latest_metrics(db: Session = Depends(get_db)):
    # This endpoint doesn't use the processor, so no changes are needed.
    latest_log = db_service.get_latest_log(db)
    if latest_log:
        return { "vehicle_count": latest_log.vehicle_count, "density": latest_log.density }
    return {"vehicle_count": 0, "density": "low"}

@router.get("/historical-data")
def get_historical_data(db: Session = Depends(get_db)):
    # This endpoint doesn't use the processor, so no changes are needed.
    logs = db_service.get_historical_logs(db)
    # Use a list comprehension for a cleaner look
    return [{"timestamp": log.timestamp.strftime("%H:%M"), "vehicle_count": log.vehicle_count} for log in logs]


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Handles video file uploads. Saves the file and returns its path.
    """
    # Sanitize filename for security
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOADS_DIR, safe_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"status": "success", "file_path": file_path}


@router.post("/source")
def set_video_source(request: Request, config: dict = Body(...)):
    """
    Switches the video source for the processor.
    Example body:
    { "type": "file", "path": "path/to/video.mp4" }
    { "type": "camera", "path": 0 }  // 0 for default webcam
    """
    processor = request.app.state.traffic_processor
    
    # Update the global state
    request.app.state.video_source = config
    
    # CRITICAL: Reset the processor for the new source
    processor.reset()
    
    return {"status": "success", "message": f"Video source changed to {config['type']}"}


@router.get("/vehicle-distribution")
def get_vehicle_distribution_data(db: Session = Depends(get_db)):
    """
    Provides data for the vehicle distribution pie chart.
    """
    return db_service.get_vehicle_distribution(db, hours=24)