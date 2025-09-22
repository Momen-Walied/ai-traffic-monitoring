# --- IMPORTS ---
import cv2
import numpy as np
import math
import time
from collections import defaultdict, deque
from typing import Optional, List, Tuple, Dict
import threading
import onnxruntime as ort # <<< Replaces ultralytics

# Re-ID still uses PyTorch, so these are still needed
import torch
import torchvision.models as models
import torchvision.transforms as T

from app.db.session import SessionLocal
# Make sure your service and models imports are correct for your project structure
from app.services import database_service as db_service
from app.db import models as dbmodels
from scipy.optimize import linear_sum_assignment


# ----------------- CONFIG -----------------
CONFIDENCE_THRESHOLD = 0.45 # Lower for more detections, higher for fewer but more confident ones
NMS_IOU_THRESHOLD = 0.5    # Threshold for removing overlapping boxes

# Your existing config constants
USE_GPU = True
REID_SIMILARITY_THRESHOLD = 0.75
REID_RETENTION_SEC = 60 * 5
TRACK_MAX_MISSED = 30 # A vehicle is "lost" if not seen for this many frames
DEBOUNCE_FRAMES = 2
SNAPSHOT_FRAMES = 90
METERS_PER_PIXEL = 0.02
SPEED_UNIT = "kmh"


# ----------------- Helpers -----------------
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1 is None or v2 is None: return -1.0
    v1 = v1.astype(np.float32)
    v2 = v2.astype(np.float32)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0: return -1.0
    return float(np.dot(v1, v2) / denom)


def color_stat_embedding(crop: np.ndarray) -> Optional[np.ndarray]:
    """Very cheap embedding: mean color per channel, normalized."""
    if crop is None or crop.size == 0:
        return None
    try:
        small = cv2.resize(crop, (64, 128))
    except Exception:
        return None
    arr = small.astype(np.float32) / 255.0
    emb = arr.mean(axis=(0, 1))  # shape (3,)
    norm = np.linalg.norm(emb)
    if norm == 0: return None
    return (emb / norm).astype(np.float32)


class CNNReID:
    """ResNet50 backbone (global feature). Optional but more accurate for ReID."""
    def __init__(self, device: str = "cpu"):
        self.device = device
        backbone = models.resnet50(pretrained=True)
        modules = list(backbone.children())[:-1]
        self.net = torch.nn.Sequential(*modules).to(self.device).eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def encode(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        try:
            x = self.transform(crop).unsqueeze(0).to(self.device)
        except Exception:
            return None
        with torch.no_grad():
            feat = self.net(x)              # (1, 2048, 1, 1)
            feat = feat.reshape(1, -1)     # (1, 2048)
            feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-6)
            return feat.cpu().numpy().flatten().astype(np.float32)


# ----------------- Processor -----------------
class TrafficProcessor:
    # Class names for the default COCO model.
    # The ONNX file doesn't store these, so we must define them.
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    VEHICLE_TYPES = {"car", "truck", "bus", "motorcycle"}

    def __init__(
        self,
        model_path: str = "yolov8n.onnx", # <<< Default to .onnx file
        use_cnn_reid: bool = True,
        meters_per_pixel: float = METERS_PER_PIXEL,
    ):
        print("[INFO] Initializing ONNX-based TrafficProcessor...")

        # --- ONNX Runtime Session Initialization ---
        available_providers = ort.get_available_providers()
        chosen_provider = 'CPUExecutionProvider'
        if USE_GPU and 'CUDAExecutionProvider' in available_providers:
            chosen_provider = 'CUDAExecutionProvider'
        else:
            print("[WARNING] GPU requested or CUDA not available. Falling back to CPU.")
        
        print(f"[INFO] Using ONNX Runtime with provider: {chosen_provider}")
        self.session = ort.InferenceSession(model_path, providers=[chosen_provider])
        
        # --- Interrogate the model to get input/output details ---
        self.model_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in self.model_inputs]
        self.input_shape = self.model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        self.model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in self.model_outputs]
        # --- End ONNX Setup ---

        self.use_cnn_reid = use_cnn_reid
        self.meters_per_pixel = float(meters_per_pixel)
        
        reid_device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
        self.cnn_reid = CNNReID(reid_device) if self.use_cnn_reid else None

        self.reid_lock = threading.Lock()
        self.reset()
        
    def reset(self):
        print("[INFO] Resetting TrafficProcessor state.")
        self.track_history = defaultdict(list)
        self.track_last_seen = {}
        self.track_missed = defaultdict(int)
        with self.reid_lock:
            self.reid_store = deque()
        self.counted_track_ids = set()
        self.total_vehicle_count = 0
        self.track_cross_buffer = defaultdict(int)
        self.dynamic_line_y = None
        self.center_y_samples = []
        self.line_calibrated = False
        self.buffer_new_detected = 0
        self.frame_idx = 0
        self.vehicle_history_buffer = []
        self.detected_vehicles_buffer = []

        # Simple Tracker State
        self.next_track_id = 0
        self.tracked_objects: Dict[int, Dict] = {} # {track_id: {'box': [x1,y1,x2,y2], 'missed': 0, 'class_name': 'car'}}

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Prepares a frame for ONNX inference."""
        img_h, img_w, _ = frame.shape
        scale = min(self.input_height / img_h, self.input_width / img_w)
        resized_w, resized_h = int(img_w * scale), int(img_h * scale)
        resized_img = cv2.resize(frame, (resized_w, resized_h))
        
        padded_img = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded_img[:resized_h, :resized_w] = resized_img
        
        image_data = np.transpose(padded_img, (2, 0, 1)) # HWC to CHW
        image_data = image_data[np.newaxis, :, :, :].astype(np.float32) / 255.0 # Add batch dim and normalize
        
        return image_data, scale

    def _postprocess(self, outputs: np.ndarray, scale: float, original_shape: Tuple[int, int]) -> List[Tuple[List[int], float, str]]:
        """Decodes raw ONNX output, applies NMS, and scales boxes back to original frame size."""
        predictions = np.squeeze(outputs).T
        
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > CONFIDENCE_THRESHOLD, :]
        scores = scores[scores > CONFIDENCE_THRESHOLD]
        
        if len(predictions) == 0:
            return []
            
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # Get boxes and convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        box_preds = predictions[:, :4]
        cx, cy, w, h = box_preds[:, 0], box_preds[:, 1], box_preds[:, 2], box_preds[:, 3]
        x1 = (cx - w / 2) / scale
        y1 = (cy - h / 2) / scale
        x2 = (cx + w / 2) / scale
        y2 = (cy + h / 2) / scale
        
        boxes = np.array([[x1[i], y1[i], x2[i], y2[i]] for i in range(len(x1))])
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD)
        
        results = []
        for i in indices:
            box = boxes[i].astype(int).tolist()
            score = scores[i]
            class_id = class_ids[i]
            class_name = self.COCO_CLASSES[class_id]
            # Filter for vehicle types
            if class_name in self.VEHICLE_TYPES:
                results.append((box, score, class_name))
        return results

    def _simple_tracker_update(self, detections: List[Tuple[List[int], float, str]]):
        """
        Updates tracked objects using the Hungarian algorithm for optimal assignment.
        This version uses a more explicit state tracking to prevent KeyErrors.
        """
        # --- Handle edge case of no detections ---
        if not detections:
            for track_id in self.tracked_objects:
                self.tracked_objects[track_id]['missed'] += 1
            # Prune any tracks that have been missed for too long
            self._prune_lost_tracks()
            return

        # --- Handle edge case of no currently tracked objects ---
        if not self.tracked_objects:
            for box, score, class_name in detections:
                self._register_new_object(box, class_name)
            return

        # --- Main Matching Logic ---
        track_ids = list(self.tracked_objects.keys())
        last_centroids = np.array([obj['centroid'] for obj in self.tracked_objects.values()])
        
        detection_centroids = np.array([((det[0][0] + det[0][2]) // 2, (det[0][1] + det[0][3]) // 2) for det in detections])

        # Build the cost matrix based on Euclidean distance
        cost_matrix = np.zeros((len(last_centroids), len(detection_centroids)))
        for i in range(len(last_centroids)):
            for j in range(len(detection_centroids)):
                cost_matrix[i, j] = math.hypot(last_centroids[i][0] - detection_centroids[j][0], last_centroids[i][1] - detection_centroids[j][1])

        # Use the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # --- State Update Logic (REFINED) ---
        MAX_MATCH_DISTANCE = 75.0
        
        # Keep track of which indices have been used
        matched_track_indices = set()
        matched_detection_indices = set()

        # Process the optimal assignments found by the algorithm
        for r, c in zip(row_ind, col_ind):
            # If the cost is too high, treat it as a non-match
            if cost_matrix[r, c] > MAX_MATCH_DISTANCE:
                continue

            track_id = track_ids[r]
            box, score, class_name = detections[c]

            # Update the matched track
            self.tracked_objects[track_id]['box'] = box
            self.tracked_objects[track_id]['centroid'] = detection_centroids[c]
            self.tracked_objects[track_id]['missed'] = 0
            self.tracked_objects[track_id]['class_name'] = class_name
            
            # Mark these indices as used
            matched_track_indices.add(r)
            matched_detection_indices.add(c)

        # --- Handle Unmatched Tracks and Detections ---

        # Any track index not in our matched set is a "missed" track
        for i in range(len(track_ids)):
            if i not in matched_track_indices:
                track_id = track_ids[i]
                self.tracked_objects[track_id]['missed'] += 1

        # Any detection index not in our matched set is a new object
        for i in range(len(detections)):
            if i not in matched_detection_indices:
                box, score, class_name = detections[i]
                self._register_new_object(box, class_name)
        
        # --- Prune lost tracks at the very end ---
        self._prune_lost_tracks()

    def _register_new_object(self, box: List[int], class_name: str):
        """Helper function to register a new tracked object."""
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        self.tracked_objects[self.next_track_id] = {
            'box': box, 'missed': 0, 'class_name': class_name, 'centroid': (cx, cy)
        }
        self.next_track_id += 1

    def _prune_lost_tracks(self):
        """Helper function to remove tracks that have been missed for too long."""
        lost_ids = [tid for tid, obj in self.tracked_objects.items() if obj['missed'] > TRACK_MAX_MISSED]
        for tid in lost_ids:
            if tid in self.tracked_objects:
                del self.tracked_objects[tid]

    # Your original helper methods (_store_reid, _find_similar_canonical, _compute_motion, etc.)
    # remain largely unchanged. Just ensure they are here. I've included them for completeness.
    def _store_reid(self, canonical_id: str, emb: np.ndarray):
        if emb is None: return
        ts = time.time()
        with self.reid_lock:
            self.reid_store.append((ts, canonical_id, emb))
            cutoff = ts - REID_RETENTION_SEC
            while self.reid_store and self.reid_store[0][0] < cutoff:
                self.reid_store.popleft()

    def _find_similar_canonical(self, emb: np.ndarray) -> Optional[str]:
        if emb is None: return None
        best_id, best_sim = None, -1.0
        with self.reid_lock:
            store_copy = list(self.reid_store)
        for ts, cid, cemb in reversed(store_copy):
            sim = cosine_similarity(emb, cemb)
            if sim > best_sim:
                best_sim, best_id = sim, cid
            if best_sim >= REID_SIMILARITY_THRESHOLD:
                return best_id
        return None

    def _cleanup_tracks(self, current_ts: float):
        # This function might need adjustment based on the simple tracker's state
        pass # The simple tracker prunes itself, so this might not be needed.

    def _compute_motion(self, points: List[tuple]) -> Tuple[Optional[float], Optional[str]]:
        if len(points) < 2: return None, None
        (x1, y1, t1), (x2, y2, t2) = points[-2], points[-1]
        dt = max(1e-3, t2 - t1)
        dist_pixels = math.hypot(x2 - x1, y2 - y1)
        dist_meters = dist_pixels * self.meters_per_pixel
        speed_mps = dist_meters / dt
        speed_kmh = speed_mps * 3.6
        dx, dy = x2 - x1, y2 - y1
        direction = "right" if dx > 0 else "left" if abs(dx) > abs(dy) else "down" if dy > 0 else "up"
        return speed_kmh if SPEED_UNIT == "kmh" else speed_mps, direction

    def _extract_embedding(self, frame: np.ndarray, box: List[int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = box
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0: return None
        if self.cnn_reid:
            return self.cnn_reid.encode(crop)
        return color_stat_embedding(crop)

    def _buffer_vehicle_data(self, canonical_id: str, embedding: Optional[np.ndarray], speed: Optional[float], direction: Optional[str], vehicle_type: str, roi_id: Optional[int] = None, trajectory_points: Optional[List[tuple]] = None):
        vehicle_data = {
            "vehicle_id": str(canonical_id), "vehicle_type": vehicle_type, "roi_id": str(roi_id) if roi_id else None,
            "speed": float(speed) if speed is not None else None, 
            "direction": str(direction) if direction is not None else None,
            "status": "counted", 
            
            # --- THIS IS THE LINE TO FIX ---
            # OLD, AMBIGUOUS WAY:
            # "embedding": embedding.tolist() if embedding else None,
            
            # NEW, EXPLICIT WAY:
            "embedding": embedding.tolist() if embedding is not None else None,
            
            "trajectory": [(float(px), float(py)) for (px, py, pt) in trajectory_points[-10:]] if trajectory_points else []
        }
        self.vehicle_history_buffer.append(vehicle_data)

    # ---- main processing generator ----
    def process_video(self, video_path: str, auto_line: bool = True, roi_vertices_frac=None):
        if roi_vertices_frac is None:
            # Default ROI covers the bottom half of the screen
            roi_vertices_frac = np.array([[0, 0.5], [1, 0.5], [1, 1], [0, 1]], np.float32)

        # Attempt to open the video source (file or camera device ID)
        try:
            # If video_path is a number (e.g., 0 for webcam), it needs to be an int
            source_to_open = int(video_path) if str(video_path).isdigit() else video_path
            cap = cv2.VideoCapture(source_to_open)
            if not cap.isOpened():
                raise IOError(f"Cannot open video source: {video_path}")
        except Exception as e:
            print(f"[ERROR] Failed to open video capture: {e}")
            return # Exit the generator if the source can't be opened

        # Get video properties
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define ROI in pixel coordinates
        roi_vertices_px = (roi_vertices_frac * np.array([frame_w, frame_h])).astype(np.int32)
        
        # Constants for dynamic line calibration
        WARMUP_FRAMES, MIN_CENTER_SAMPLES, PERCENTILE_FOR_LINE = 60, 30, 55
        LINE_OFFSET, LINE_X_MARGIN_PCT, FALLBACK_LINE_RATIO = 20, 0.05, 0.60

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # --- GRACEFUL HANDLING OF VIDEO END / STREAM DISCONNECT ---
                    print("[INFO] End of video stream reached.")
                    
                    # If the source was a file (not a camera), try to loop it
                    if not str(video_path).isdigit():
                        # 1. Force a save of any pending data in the buffer before looping.
                        if self.vehicle_history_buffer:
                            print("[INFO] Saving final buffered data before loop...")
                            # Calculate average density for the final segment
                            avg_vehicles_on_screen = 0.0
                            if self.detected_vehicles_buffer:
                                avg_vehicles_on_screen = sum(self.detected_vehicles_buffer) / len(self.detected_vehicles_buffer)
                            
                            if avg_vehicles_on_screen > 8: period_density = "high"
                            elif avg_vehicles_on_screen > 3: period_density = "medium"
                            else: period_density = "low"
                            
                            db = SessionLocal()
                            try:
                                current_roi_id = self.vehicle_history_buffer[0].get("roi_id", "1")
                                new_log = dbmodels.TrafficLog(
                                    vehicle_count=len(self.vehicle_history_buffer),
                                    density=period_density,
                                    avg_vehicles_on_screen=avg_vehicles_on_screen,
                                    roi_id=current_roi_id
                                )
                                db.add(new_log)
                                db.flush()
                                for v_data in self.vehicle_history_buffer:
                                    vh = dbmodels.VehicleHistory(
                                        vehicle_id=v_data["vehicle_id"], vehicle_type=v_data["vehicle_type"],
                                        traffic_log_id=new_log.id, roi_id=v_data["roi_id"],
                                        speed=v_data["speed"], direction=v_data["direction"],
                                        status=v_data["status"], embedding=v_data["embedding"]
                                    )
                                    db.add(vh)
                                    db.flush()
                                    for px, py in v_data["trajectory"]:
                                        db.add(dbmodels.VehicleTrajectory(
                                            vehicle_history_id=vh.id, vehicle_id=v_data["vehicle_id"], x=px, y=py
                                        ))
                                db.commit()
                                print(f"[DB] SAVED End-of-Stream Snapshot. Count: {len(self.vehicle_history_buffer)}, Density: {period_density}")
                            except Exception as e:
                                print(f"[DB ERROR] Final save failed: {e}")
                                db.rollback()
                            finally:
                                db.close()
                        
                        # 2. Reset the entire processor state for a clean start
                        print("[INFO] Video file looped -> resetting full processor state.")
                        self.reset() 

                        # 3. Seek the video back to the beginning and continue the loop
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.1)
                        continue
                    else:
                        # If it's a camera stream that ended, just break the loop
                        print("[INFO] Live camera stream ended.")
                        break

                # --- Main processing logic for a valid frame ---
                t_now = time.time()
                self.frame_idx += 1
                annotated = frame.copy()

                # --- ONNX Inference Pipeline ---
                input_tensor, scale = self._preprocess(frame)
                outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
                detections = self._postprocess(outputs[0], scale, (frame_h, frame_w))
                
                # --- Tracker Update ---
                self._simple_tracker_update(detections)
                self.detected_vehicles_buffer.append(len(self.tracked_objects))
                
                # --- Auto-line Calibration ---
                if auto_line and not self.line_calibrated:
                    self.center_y_samples.extend([obj['centroid'][1] for obj in self.tracked_objects.values()])
                    if self.frame_idx >= WARMUP_FRAMES and len(self.center_y_samples) >= MIN_CENTER_SAMPLES:
                        self.dynamic_line_y = int(np.percentile(self.center_y_samples, PERCENTILE_FOR_LINE))
                        self.line_calibrated = True
                        self.center_y_samples.clear()
                        print(f"[AUTO-LINE] calibrated y={self.dynamic_line_y}")

                # --- Define Counting Lines ---
                line_y1 = self.dynamic_line_y if self.dynamic_line_y is not None else int(frame_h * FALLBACK_LINE_RATIO)
                line_y2 = line_y1 + LINE_OFFSET
                LINE_START1, LINE_END1 = (int(frame_w * LINE_X_MARGIN_PCT), line_y1), (int(frame_w * (1 - LINE_X_MARGIN_PCT)), line_y1)

                # --- Process Each Tracked Object ---
                for track_id, obj in list(self.tracked_objects.items()):
                    box = obj['box']
                    cx, cy = obj['centroid']
                    vehicle_type = obj['class_name']
                    
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"ID:{track_id} {vehicle_type}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    self.track_history[track_id].append((cx, cy, t_now))
                    self.track_history[track_id] = self.track_history[track_id][-120:]
                    self.track_last_seen[track_id] = t_now

                    # --- Vehicle Counting Logic ---
                    if self.line_calibrated and len(self.track_history[track_id]) > 1:
                        prev_y = self.track_history[track_id][-2][1]
                        curr_y = cy
                        
                        crossed_line = (prev_y < line_y1 and curr_y >= line_y1) or (prev_y > line_y1 and curr_y <= line_y1)

                        if crossed_line:
                            self.track_cross_buffer[track_id] += 1
                        else:
                            if not (min(line_y1, line_y2) < curr_y < max(line_y1, line_y2)):
                                self.track_cross_buffer[track_id] = 0
                        
                        if self.track_cross_buffer[track_id] >= DEBOUNCE_FRAMES:
                            if track_id not in self.counted_track_ids:
                                self.counted_track_ids.add(track_id)
                                self.total_vehicle_count += 1
                                self.buffer_new_detected += 1
                                
                                speed, direction = self._compute_motion(self.track_history[track_id])
                                emb = self._extract_embedding(frame, box)
                                self._store_reid(str(track_id), emb)
                                self._buffer_vehicle_data(track_id, emb, speed, direction, vehicle_type, roi_id=1, trajectory_points=self.track_history[track_id])
                                
                                print(f"[COUNT] frame={self.frame_idx} track={track_id} type={vehicle_type} total={self.total_vehicle_count} speed={speed if speed is not None else 0:.2f}")

                                # EVEN BETTER, MORE READABLE VERSION (Recommended):
                                speed_to_print = speed if speed is not None else 0
                                print(f"[COUNT] frame={self.frame_idx} track={track_id} type={vehicle_type} total={self.total_vehicle_count} speed={speed_to_print:.2f}")
                                # Visual flash for count
                                cv2.line(annotated, LINE_START1, LINE_END1, (0,0,255), 4)

                            self.track_cross_buffer[track_id] = 0

                # --- Periodic Snapshot Saving to Database ---
                if self.frame_idx % SNAPSHOT_FRAMES == 0 and self.vehicle_history_buffer:
                    avg_vehicles_on_screen = sum(self.detected_vehicles_buffer) / len(self.detected_vehicles_buffer) if self.detected_vehicles_buffer else 0
                    if avg_vehicles_on_screen > 8: period_density = "high"
                    elif avg_vehicles_on_screen > 3: period_density = "medium"
                    else: period_density = "low"

                    db = SessionLocal()
                    try:
                        current_roi_id = self.vehicle_history_buffer[0].get("roi_id", "1")
                        new_log = dbmodels.TrafficLog(
                            vehicle_count=len(self.vehicle_history_buffer), density=period_density,
                            avg_vehicles_on_screen=avg_vehicles_on_screen, roi_id=current_roi_id
                        )
                        db.add(new_log)
                        db.flush()
                        for v_data in self.vehicle_history_buffer:
                            vh = dbmodels.VehicleHistory(
                                vehicle_id=v_data["vehicle_id"], vehicle_type=v_data["vehicle_type"],
                                traffic_log_id=new_log.id, roi_id=v_data["roi_id"],
                                speed=v_data["speed"], direction=v_data["direction"],
                                status=v_data["status"], embedding=v_data["embedding"]
                            )
                            db.add(vh)
                            db.flush()
                            for px, py in v_data["trajectory"]:
                                db.add(dbmodels.VehicleTrajectory(
                                    vehicle_history_id=vh.id, vehicle_id=v_data["vehicle_id"], x=px, y=py
                                ))
                        db.commit()
                        print(f"[DB] SAVED Snapshot. Count: {len(self.vehicle_history_buffer)}, Density: {period_density}")
                    except Exception as e:
                        print(f"[DB ERROR] Snapshot save failed: {e}")
                        db.rollback()
                    finally:
                        db.close()
                    
                    # Clear buffers after saving
                    self.vehicle_history_buffer.clear()
                    self.detected_vehicles_buffer.clear()
                    self.buffer_new_detected = 0

                # --- Draw Overlays on Frame ---
                cv2.polylines(annotated, [roi_vertices_px], isClosed=True, color=(255, 255, 0), thickness=2)
                if self.line_calibrated:
                    cv2.line(annotated, LINE_START1, LINE_END1, (0, 255, 0), 2)
                
                cv2.putText(annotated, f"Total Count: {self.total_vehicle_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(annotated, f"Buffered: {self.buffer_new_detected}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
                
                # --- Yield Frame for Streaming ---
                (flag, encodedImage) = cv2.imencode(".jpg", annotated)
                if not flag:
                    continue
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

        finally:
            # This block runs when the generator is closed
            cap.release()
            print("[INFO] process_video finished, camera released.")