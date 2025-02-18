# filepath: /c:/Users/ethan/GIT/GIT/Python/wood_counter/wood_counter_train.py
from ultralytics import YOLO
from roboflow import Roboflow
import os

# Initialize Roboflow
rf = Roboflow(api_key="083dTBfHrUuMsxbr7BFA")
project = rf.workspace("jetcounterdemo").project("logs_half")
version = project.version(2)
dataset = version.download("yolov5")

# Path to the data.yaml file
data_yaml_path = 'C:/Users/ethan/GIT/GIT/Python/wood_counter/data.yaml'

# Check if the data.yaml file exists
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"'{data_yaml_path}' does not exist. Please check the path.")

# Initialize a new YOLOv5 model architecture
model = YOLO('yolov5s.yaml')  # Use the YOLOv5s architecture

# Train the model on your custom dataset
model.train(data=data_yaml_path, epochs=3, batch=16, imgsz=640)