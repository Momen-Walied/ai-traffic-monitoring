from sqlalchemy import Column, Integer, String, DateTime, func, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class TrafficLog(Base):
    __tablename__ = "traffic_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    vehicle_count = Column(Integer, nullable=False)
    density = Column(String, nullable=False)
    avg_vehicles_on_screen = Column(Float, nullable=True)

    # ✅ اربط بالـ ROI (String roi_id)
    roi_id = Column(String, ForeignKey("rois.roi_id", ondelete="CASCADE"), index=True)

    vehicles = relationship("VehicleHistory", back_populates="traffic_log")
    roi = relationship("ROI", back_populates="traffic_logs")  # بدل backref
    


class VehicleHistory(Base):
    __tablename__ = "vehicle_history"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    vehicle_type = Column(String, nullable=True, index=True)

    traffic_log_id = Column(Integer, ForeignKey("traffic_logs.id", ondelete="CASCADE"))
    traffic_log = relationship("TrafficLog", back_populates="vehicles")

    # ✅ اربط برضو بالـ ROI
    roi_id = Column(String, ForeignKey("rois.roi_id", ondelete="CASCADE"), index=True)
    roi = relationship("ROI", back_populates="vehicles")

    speed = Column(Float)
    direction = Column(String)
    status = Column(String, default="passing")

    trajectory = relationship("VehicleTrajectory", back_populates="vehicle")

    embedding = Column(Vector(2048))  # 512 dimension vector


class VehicleTrajectory(Base):
    __tablename__ = "vehicle_trajectory"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(String, index=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    vehicle_history_id = Column(Integer, ForeignKey("vehicle_history.id", ondelete="CASCADE"))
    vehicle = relationship("VehicleHistory", back_populates="trajectory")


class ROI(Base):
    __tablename__ = "rois"

    id = Column(Integer, primary_key=True, index=True)
    roi_id = Column(String, unique=True, index=True)
    name = Column(String)
    description = Column(String)
    coordinates = Column(String)  # JSON string

    traffic_logs = relationship("TrafficLog", back_populates="roi", cascade="all, delete-orphan")
    vehicles = relationship("VehicleHistory", back_populates="roi", cascade="all, delete-orphan")
