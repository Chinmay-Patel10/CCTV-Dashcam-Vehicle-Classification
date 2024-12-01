import torch
from PIL import Image
import os
from model_inference import predict_top_k, test_transform

# Load the YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()  # Set model to evaluation mode

# Set global confidence and IoU thresholds
yolo_model.conf = 0.2
yolo_model.iou = 0.5

def detect_and_classify(image_path):
    image = Image.open(image_path).convert('RGB')
    
    # Run YOLOv5 model
    results = yolo_model(image)
    
    # Define the path to save the boxed image
    boxed_image_path = os.path.join('static', 'boxed_' + os.path.basename(image_path))
    
    # Save the image with bounding boxes
    results.save(save_dir='static')  # Save results in the 'static' folder
    
    # Look for the generated image with bounding boxes in the save directory
    # YOLOv5 generally names it based on the original filename
    expected_boxed_image_name = os.path.basename(image_path)  # The boxed image should have the same base name
    boxed_image_full_path = os.path.join('static', expected_boxed_image_name)
    
    # Verify that the boxed image is in the correct directory
    if os.path.exists(boxed_image_full_path):
        boxed_image_path = boxed_image_full_path  # Update the path to the correct boxed image

    # Filter for vehicles and prepare vehicle detections
    vehicle_detections = []
    for *coords, conf, cls in results.xyxy[0]:  # Extract each detection
        label = yolo_model.names[int(cls)]
        if label in ["car", "truck", "bus", "motorbike"]:  # Only select vehicle classes
            vehicle_detections.append({
                "label": label,
                "confidence": conf.item(),
                "coords": coords
            })
    
    # Pass the cropped vehicle image to the classifier if any vehicle is detected
    if vehicle_detections:
        top_predictions = predict_top_k(image_path)
    else:
        top_predictions = []

    return top_predictions, boxed_image_path
