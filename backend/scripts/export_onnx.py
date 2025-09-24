from ultralytics import YOLO

# Load your trained PyTorch model
# If you just used the base model, it's 'yolov8n.pt'
model = YOLO("yolov8l.pt") 

# Export the model to ONNX format
# You can specify imgsz (image size) for a static-sized model, which is faster.
# Let's assume a common input size like 640x640.
model.export(format="onnx", imgsz=640, opset=17)

print("Model exported to yolov8n.onnx successfully!")