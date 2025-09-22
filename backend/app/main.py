from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import settings
from app.services.video_processor import TrafficProcessor

# Import these for the ROI fix
from app.db.session import SessionLocal
from app.db import models as dbmodels

# --- 1. Create the App Instance ---
app = FastAPI(title=settings.PROJECT_NAME)

# --- 2. Define the Lifespan Events (Modern Way) ---
DEFAULT_VIDEO_PATH = r"C:\Users\momen\Downloads\2103099-uhd_3840_2160_30fps.mp4"

@app.on_event("startup")
def startup_event():
    """
    On application startup:
    1. Ensure the default ROI exists in the database.
    2. Create the single TrafficProcessor instance and attach it to app.state.
    """
    print("[INFO] Application starting up...")

    # --- Robust Fix: Create default ROI if it doesn't exist ---
    db = SessionLocal()
    try:
        existing_roi = db.query(dbmodels.ROI).filter(dbmodels.ROI.roi_id == '1').first()
        if not existing_roi:
            print("[INFO] Default ROI '1' not found. Creating it...")
            default_roi = dbmodels.ROI(
                roi_id='1',
                name='Default Camera ROI',
                description='Auto-created default ROI on startup',
                coordinates='[[0,0.5],[1,0.5],[1,1],[0,1]]' # Example coordinates
            )
            db.add(default_roi)
            db.commit()
            print("[INFO] Default ROI '1' created successfully.")
        else:
            print("[INFO] Default ROI '1' already exists.")
    finally:
        db.close()
    # --- End of ROI Fix ---

    # Create the processor and attach it DIRECTLY to app.state
    processor = TrafficProcessor(model_path="yolov8l.onnx") 
    app.state.traffic_processor = processor
    app.state.video_source = {
        "type": "file",
        "path": DEFAULT_VIDEO_PATH
    }
    print("[INFO] TrafficProcessor instance created and attached to app state.")


@app.on_event("shutdown")
def shutdown_event():
    """
    On application shutdown, add cleanup logic here if needed.
    """
    print("[INFO] Application shutting down...")
    # Access the processor from app.state to clean up if needed
    if hasattr(app.state, "traffic_processor"):
        del app.state.traffic_processor
        print("[INFO] Processor resources cleaned up.")


# --- 3. Add Middleware and Routers ---
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

# --- 4. Define Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "🚦 AI Traffic Monitoring API is running"}