from sqlalchemy.orm import Session
from app.db import models
from datetime import datetime, timedelta
from collections import Counter
from sqlalchemy import func

def save_traffic_log(db: Session, vehicle_count: int, density: str):
    db_log = models.TrafficLog(vehicle_count=vehicle_count, density=density)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_latest_log(db: Session):
    return db.query(models.TrafficLog).order_by(models.TrafficLog.timestamp.desc()).first()

def get_historical_logs(db: Session):
    """Retrieves all raw traffic logs from the last 24 hours."""
    time_24_hours_ago = datetime.now() - timedelta(hours=24)
    return db.query(models.TrafficLog).filter(models.TrafficLog.timestamp >= time_24_hours_ago).order_by(models.TrafficLog.timestamp.asc()).all()

def get_aggregated_historical_logs(db: Session):
    """
    Retrieves and aggregates logs from the last hour, summarizing
    data by the minute for cleaner LLM context.
    """
    time_1_hour_ago = datetime.now() - timedelta(hours=1)
    
    # Query to get total vehicle count and all density values per minute
    results = db.query(
        func.date_trunc('minute', models.TrafficLog.timestamp).label('minute'),
        func.sum(models.TrafficLog.vehicle_count).label('total_vehicles'),
        func.array_agg(models.TrafficLog.density).label('densities')
    ).filter(
        models.TrafficLog.timestamp >= time_1_hour_ago
    ).group_by(
        'minute'
    ).order_by(
        'minute'
    ).all()

    # Process results to find the most common density for each minute
    aggregated_data = []
    for row in results:
        # The 'densities' field is a list of strings like ['low', 'medium', 'medium']
        # Counter finds the most common one.
        most_common_density = Counter(row.densities).most_common(1)[0][0] if row.densities else 'unknown'
        aggregated_data.append({
            "timestamp": row.minute,
            "vehicle_count": row.total_vehicles,
            "density": most_common_density
        })
        
    return aggregated_data

def get_vehicle_distribution(db: Session, hours: int = 24):
    """
    Counts vehicle types over a given period.
    """
    time_threshold = datetime.utcnow() - timedelta(hours=hours)
    
    # Query to group by vehicle_type and count them
    distribution = (
        db.query(
            models.VehicleHistory.vehicle_type,
            func.count(models.VehicleHistory.id).label("count")
        )
        .filter(models.VehicleHistory.timestamp >= time_threshold)
        .group_by(models.VehicleHistory.vehicle_type)
        .order_by(func.count(models.VehicleHistory.id).desc())
        .all()
    )
    
    # Format the result for the frontend
    return [{"name": v_type if v_type else "unknown", "value": count} for v_type, count in distribution]