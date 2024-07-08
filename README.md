# Real-Time Object Detection with YOLOv8 and OpenCV

This repository contains a project for real-time object detection using the YOLOv8 model and OpenCV. The project demonstrates how to use a pre-trained YOLO model to detect various objects in a live video stream from a webcam. A special feature is implemented to highlight knives with a red bounding box.

## Project Overview

The main goal of this project is to showcase real-time object detection capabilities using the YOLOv8 model. The YOLO (You Only Look Once) model is known for its speed and accuracy in object detection tasks. This project includes the following steps:

1. **Initializing the Webcam:** Setting up the webcam and configuring the resolution.
2. **Loading the YOLO Model:** Using a pre-trained YOLOv8 model for object detection.
3. **Defining Classes:** Listing the objects that the model can detect.
4. **Capturing and Processing Video Frames:** Reading frames from the webcam and processing them with the YOLO model to detect objects.
5. **Drawing Bounding Boxes:** Highlighting detected objects with bounding boxes and displaying their class names and confidence scores.
6. **Special Handling for Knives:** Drawing a red bounding box around knives to make them easily identifiable.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv8

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MarBenitez/yolo-red-kn/yolo-knife-detection.py
   cd real-time-object-detection
