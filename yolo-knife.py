from ultralytics import YOLO
import cv2
import math

# Initialize webcam and establish image resolution
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Resolution width
cap.set(4, 720)   # Resolution height

# Load YOLO model
model = YOLO('yolo-weights/yolov8n.pt')

# Define the classes we want to detect
classNames = [
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

# Capture loop
while True:
    success, img = cap.read()  # Read the image from the camera
    results = model(img, stream=True)  # Send the image to YOLO to detect objects

    # Loop over the detected objects
    for r in results:
        boxes = r.boxes  # Get the bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get the box coordinates
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Convert to int

            # Detect the class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            confidence = math.ceil(box.conf[0] * 100)  # Round the confidence to an integer

            # Draw the bounding box
            color = (255, 0, 0)  # Default color: blue
            if class_name == 'knife':
                color = (0, 0, 255)  # Red color for knives
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # Print the class name and confidence
            print(f'{class_name}: {confidence}%')

            # Draw the class name and confidence
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f'{class_name}: {confidence}%', org, font, 1, color, 2)

    # Create a window to show the image
    cv2.imshow('webcam', img)

    # Exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
