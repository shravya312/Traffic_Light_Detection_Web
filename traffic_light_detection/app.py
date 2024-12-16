from flask import Flask, request, jsonify, send_from_directory
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import io
import os

# Set the project name
PROJECT_NAME = 'traffic_light_detection'

# Initialize Flask app
app = Flask(__name__)

# Set the path to your saved model checkpoint
MODEL_PATH = 'traffic_light_detection/models/fasterrcnn_resnet50_fpn.pth'  # Change this to your model's actual path

# Initialize the Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # Initialize with no weights
model.load_state_dict(torch.load(MODEL_PATH))  # Load the saved model weights
model.eval()  # Set the model to evaluation mode

# Set the device for model inference (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the transform to convert the image into a tensor
transform = transforms.Compose([transforms.ToTensor()])

# Function to detect traffic light color
def detect_traffic_light_color(cropped_img):
    cropped_img = np.array(cropped_img)
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)

    # Color ranges for traffic lights
    green_range = ((35, 50, 50), (85, 255, 255))
    red_range_1 = ((0, 100, 100), (10, 255, 255))
    red_range_2 = ((170, 100, 100), (180, 255, 255))
    yellow_range = ((20, 100, 100), (30, 255, 255))

    green_mask = cv2.inRange(hsv_img, green_range[0], green_range[1])
    red_mask_1 = cv2.inRange(hsv_img, red_range_1[0], red_range_1[1])
    red_mask_2 = cv2.inRange(hsv_img, red_range_2[0], red_range_2[1])
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    yellow_mask = cv2.inRange(hsv_img, yellow_range[0], yellow_range[1])

    green_percentage = np.sum(green_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
    red_percentage = np.sum(red_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
    yellow_percentage = np.sum(yellow_mask) / (cropped_img.shape[0] * cropped_img.shape[1])

    if green_percentage > red_percentage and green_percentage > yellow_percentage:
        return "Green"
    elif red_percentage > yellow_percentage:
        return "Red"
    else:
        return "Yellow"

@app.route('/detect', methods=['POST'])
def detect_traffic_lights():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image = Image.open(io.BytesIO(file.read()))
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    threshold = 0.5
    filtered_boxes = boxes[scores > threshold]
    filtered_scores = scores[scores > threshold]

    detection_results = []
    for box, score in zip(filtered_boxes, filtered_scores):
        cropped_img = image.crop((box[0], box[1], box[2], box[3]))
        light_color = detect_traffic_light_color(cropped_img)
        detection_results.append({
            "box": box.tolist(),
            "score": float(score),
            "light_color": light_color
        })

    return jsonify(detection_results)

@app.route('/')
def serve_frontend():
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
