# from flask import Flask, request, jsonify, send_from_directory,render_template
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import cv2
# import io
# import os

# # Set the project name
# PROJECT_NAME = 'traffic_light_detection'

# # Initialize Flask app
# app = Flask(__name__)

# #Set the path to your saved model checkpoint
# MODEL_PATH = 'traffic_light_detection/models/fasterrcnn_resnet50_fpn.pth'  # Change this to your model's actual path

# # Initialize the Faster R-CNN model
# model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # Initialize with no weights
# model.load_state_dict(torch.load(MODEL_PATH))  # Load the saved model weights
# model.eval()  # Set the model to evaluation mode

# # Set the device for model inference (CUDA if available, else CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define the transform to convert the image into a tensor
# transform = transforms.Compose([transforms.ToTensor()])

# # Function to detect traffic light color
# def detect_traffic_light_color(cropped_img):
#     cropped_img = np.array(cropped_img)
#     hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)

#     # Color ranges for traffic lights
#     green_range = ((35, 50, 50), (85, 255, 255))
#     red_range_1 = ((0, 100, 100), (10, 255, 255))
#     red_range_2 = ((170, 100, 100), (180, 255, 255))
#     yellow_range = ((20, 100, 100), (30, 255, 255))

#     green_mask = cv2.inRange(hsv_img, green_range[0], green_range[1])
#     red_mask_1 = cv2.inRange(hsv_img, red_range_1[0], red_range_1[1])
#     red_mask_2 = cv2.inRange(hsv_img, red_range_2[0], red_range_2[1])
#     red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
#     yellow_mask = cv2.inRange(hsv_img, yellow_range[0], yellow_range[1])

#     green_percentage = np.sum(green_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
#     red_percentage = np.sum(red_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
#     yellow_percentage = np.sum(yellow_mask) / (cropped_img.shape[0] * cropped_img.shape[1])

#     if green_percentage > red_percentage and green_percentage > yellow_percentage:
#         return "Green"
#     elif red_percentage > yellow_percentage:
#         return "Red"
#     else:
#         return "Yellow"

# @app.route('/')
# def serve_frontend():
#     return render_template('index.html')
# @app.route('/detect', methods=['POST'])
# def detect_traffic_lights():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     image = Image.open(io.BytesIO(file.read()))
#     image_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         prediction = model(image_tensor)

#     boxes = prediction[0]['boxes'].cpu().numpy()
#     labels = prediction[0]['labels'].cpu().numpy()
#     scores = prediction[0]['scores'].cpu().numpy()

#     threshold = 0.5
#     filtered_boxes = boxes[scores > threshold]
#     filtered_scores = scores[scores > threshold]

#     detection_results = []
#     for box, score in zip(filtered_boxes, filtered_scores):
#         cropped_img = image.crop((box[0], box[1], box[2], box[3]))
#         light_color = detect_traffic_light_color(cropped_img)
#         detection_results.append({
#             "box": box.tolist(),
#             "score": float(score),
#             "light_color": light_color
#         })

#     return jsonify(detection_results)

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, request, render_template, jsonify, send_file
# import os
# import torch
# import numpy as np
# from PIL import Image
# from torchvision import models, transforms
# import cv2
# import matplotlib.pyplot as plt

# Initialize Flask app
# app = Flask(__name__)

# # Load the pre-trained Faster R-CNN model
# model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define the transform to convert image into tensor
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert image to tensor and normalize
# ])

# # Function to detect the color of the traffic light
# def detect_traffic_light_color(cropped_img):
#     cropped_img = np.array(cropped_img)
#     hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
    
#     green_range = ((35, 50, 50), (85, 255, 255))
#     red_range_1 = ((0, 100, 100), (10, 255, 255))
#     red_range_2 = ((170, 100, 100), (180, 255, 255))
#     yellow_range = ((20, 100, 100), (30, 255, 255))
    
#     green_mask = cv2.inRange(hsv_img, green_range[0], green_range[1])
#     red_mask_1 = cv2.inRange(hsv_img, red_range_1[0], red_range_1[1])
#     red_mask_2 = cv2.inRange(hsv_img, red_range_2[0], red_range_2[1])
#     red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
#     yellow_mask = cv2.inRange(hsv_img, yellow_range[0], yellow_range[1])
    
#     green_percentage = np.sum(green_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
#     red_percentage = np.sum(red_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
#     yellow_percentage = np.sum(yellow_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
    
#     if green_percentage > red_percentage and green_percentage > yellow_percentage:
#         return "Green"
#     elif red_percentage > yellow_percentage:
#         return "Red"
#     else:
#         return "Yellow"

# # Route for the homepage
# @app.route('/')
# def home():
#     return render_template('index.html')  # HTML file for uploading images

# # Route for processing the uploaded image
# @app.route('/detect', methods=['POST'])
# def detect():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded.'})
    
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file.'})
    
#     # Save the uploaded image
#     image_path = os.path.join('uploads', file.filename)
#     os.makedirs('uploads', exist_ok=True)
#     file.save(image_path)
    
#     # Load and preprocess the image
#     image = Image.open(image_path)
#     image_tensor = transform(image).unsqueeze(0).to(device)
    
#     # Run inference
#     with torch.no_grad():
#         prediction = model(image_tensor)
    
#     # Process predictions
#     boxes = prediction[0]['boxes'].cpu().numpy()
#     labels = prediction[0]['labels'].cpu().numpy()
#     scores = prediction[0]['scores'].cpu().numpy()
    
#     traffic_light_class_label = 10  # Update this based on your model's labels
#     threshold = 0.5
#     traffic_light_boxes = boxes[scores > threshold]
#     traffic_light_labels = labels[scores > threshold]
#     traffic_light_scores = scores[scores > threshold]
    
#     traffic_light_boxes = traffic_light_boxes[traffic_light_labels == traffic_light_class_label]
#     traffic_light_scores = traffic_light_scores[traffic_light_labels == traffic_light_class_label]
    
#     # Visualization
#     fig, ax = plt.subplots(1, figsize=(12, 9))
#     ax.imshow(image)
    
#     for box, score in zip(traffic_light_boxes, traffic_light_scores):
#         box = box.astype(int)
#         cropped_img = image.crop((box[0], box[1], box[2], box[3]))
#         light_color = detect_traffic_light_color(cropped_img)
        
#         box_color = {"Green": "green", "Red": "red", "Yellow": "yellow"}.get(light_color, "white")
#         ax.add_patch(plt.Rectangle(
#             (box[0], box[1]),
#             box[2] - box[0],
#             box[3] - box[1],
#             fill=False,
#             color=box_color,
#             linewidth=3
#         ))
#         ax.text(
#             box[0], box[1],
#             f'Traffic Light ({light_color}) ({score:.2f})',
#             fontsize=12,
#             color='black',
#             bbox=dict(facecolor=box_color, alpha=0.5)
#         )
    
#     output_path = os.path.join('outputs', f"result_{file.filename}")
#     os.makedirs('outputs', exist_ok=True)
#     plt.savefig(output_path)
#     plt.close()
    
#     return send_file(output_path, mimetype='image/png')

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, send_file,render_template
# import torch
# from torchvision import transforms
# from PIL import Image, ImageDraw, ImageFont
# import io

# app = Flask(__name__)

# # Load the pre-trained model (e.g., Faster R-CNN)
# model = torch.load("traffic_light_detection/models/fasterrcnn_resnet50_fpn.pth")  # Adjust to your model's path
# model.eval()

# # Define the transform for the input image
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# def draw_bounding_boxes(image, predictions, confidence_threshold=0.5):
#     """
#     Draw bounding boxes and labels on the image.
#     :param image: PIL image
#     :param predictions: Model predictions (bounding boxes, labels, scores)
#     :param confidence_threshold: Confidence score threshold for filtering
#     :return: Annotated PIL image
#     """
#     draw = ImageDraw.Draw(image)
#     font = ImageFont.load_default()
    
#     # Iterate over predictions
#     for box, label, score in zip(
#         predictions["boxes"], predictions["labels"], predictions["scores"]
#     ):
#         if score >= confidence_threshold:
#             # Convert bounding box to pixel values
#             box = [int(coord) for coord in box.tolist()]
            
#             # Determine color and label for the traffic light
#             label_text = f"Traffic Light ({label}) ({score:.2f})"
#             color = "green" if label == "Green" else "yellow" if label == "Yellow" else "red"
            
#             # Draw the bounding box
#             draw.rectangle(box, outline=color, width=3)
            
#             # Draw the label background
#             label_size = draw.textsize(label_text, font=font)
#             label_background = [box[0], box[1] - label_size[1], box[0] + label_size[0], box[1]]
#             draw.rectangle(label_background, fill=color)
            
#             # Draw the label text
#             draw.text((box[0], box[1] - label_size[1]), label_text, fill="black", font=font)
    
#     return image
# @app.route('/')
# def home():
#     return render_template('index.html')  # HTML file for uploading images
# @app.route('/detect', methods=['POST'])
# def detect_objects():
#     """
#     Endpoint to detect objects and return an annotated image.
#     """
#     # Ensure an image file is provided
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400

#     # Read the image
#     file = request.files['image']
#     image = Image.open(file.stream).convert("RGB")

#     # Preprocess the image and get predictions
#     input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         predictions = model(input_tensor)[0]

#     # Draw bounding boxes and labels
#     annotated_image = draw_bounding_boxes(image, predictions)

#     # Save the result to a BytesIO stream
#     img_io = io.BytesIO()
#     annotated_image.save(img_io, 'JPEG')
#     img_io.seek(0)

#     # Return the annotated image
#     return send_file(img_io, mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, send_file, render_template
# import torch
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision import transforms
# from PIL import Image, ImageDraw, ImageFont
# import io

# app = Flask(__name__)

# # Define the model architecture
# model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=3)  # Adjust `num_classes` for your use case

# # Load the state dictionary
# model.load_state_dict(torch.load("traffic_light_detection/models/fasterrcnn_resnet50_fpn.pth"))
# model.eval()

# # Define the transform for the input image
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# def draw_bounding_boxes(image, predictions, confidence_threshold=0.5):
#     """
#     Draw bounding boxes and labels on the image.
#     :param image: PIL image
#     :param predictions: Model predictions (bounding boxes, labels, scores)
#     :param confidence_threshold: Confidence score threshold for filtering
#     :return: Annotated PIL image
#     """
#     draw = ImageDraw.Draw(image)
#     font = ImageFont.load_default()
    
#     # Iterate over predictions
#     for box, label, score in zip(
#         predictions["boxes"], predictions["labels"], predictions["scores"]
#     ):
#         if score >= confidence_threshold:
#             # Convert bounding box to pixel values
#             box = [int(coord) for coord in box.tolist()]
            
#             # Determine color and label for the traffic light
#             label_text = f"Traffic Light ({label}) ({score:.2f})"
#             color = "green" if label == 1 else "yellow" if label == 2 else "red"
            
#             # Draw the bounding box
#             draw.rectangle(box, outline=color, width=3)
            
#             # Draw the label background
#             label_size = draw.textsize(label_text, font=font)
#             label_background = [box[0], box[1] - label_size[1], box[0] + label_size[0], box[1]]
#             draw.rectangle(label_background, fill=color)
            
#             # Draw the label text
#             draw.text((box[0], box[1] - label_size[1]), label_text, fill="black", font=font)
    
#     return image

# @app.route('/')
# def home():
#     return render_template('index.html')  # HTML file for uploading images

# @app.route('/detect', methods=['POST'])
# def detect_objects():
#     """
#     Endpoint to detect objects and return an annotated image.
#     """
#     # Ensure an image file is provided
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400

#     # Read the image
#     file = request.files['image']
#     image = Image.open(file.stream).convert("RGB")

#     # Preprocess the image and get predictions
#     input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         predictions = model(input_tensor)[0]

#     # Draw bounding boxes and labels
#     annotated_image = draw_bounding_boxes(image, predictions)

#     # Save the result to a BytesIO stream
#     img_io = io.BytesIO()
#     annotated_image.save(img_io, 'JPEG')
#     img_io.seek(0)

#     # Return the annotated image
#     return send_file(img_io, mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, request, render_template, jsonify, send_file
# import os
# import torch
# import numpy as np
# from PIL import Image
# from torchvision import models, transforms
# import cv2
# import matplotlib.pyplot as plt

# # Initialize Flask app
# app = Flask(__name__)

# # Load the pre-trained Faster R-CNN model
# MODEL_PATH = 'traffic_light_detection/models/fasterrcnn_resnet50_fpn.pth'
# model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()

# # Set device for inference
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define the transform to convert image into tensor
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert image to tensor and normalize
# ])

# # Function to detect the color of the traffic light
# def detect_traffic_light_color(cropped_img):
#     cropped_img = np.array(cropped_img)
#     hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
    
#     green_range = ((35, 50, 50), (85, 255, 255))
#     red_range_1 = ((0, 100, 100), (10, 255, 255))
#     red_range_2 = ((170, 100, 100), (180, 255, 255))
#     yellow_range = ((20, 100, 100), (30, 255, 255))
    
#     green_mask = cv2.inRange(hsv_img, green_range[0], green_range[1])
#     red_mask_1 = cv2.inRange(hsv_img, red_range_1[0], red_range_1[1])
#     red_mask_2 = cv2.inRange(hsv_img, red_range_2[0], red_range_2[1])
#     red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
#     yellow_mask = cv2.inRange(hsv_img, yellow_range[0], yellow_range[1])
    
#     green_percentage = np.sum(green_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
#     red_percentage = np.sum(red_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
#     yellow_percentage = np.sum(yellow_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
    
#     if green_percentage > red_percentage and green_percentage > yellow_percentage:
#         return "Green"
#     elif red_percentage > yellow_percentage:
#         return "Red"
#     else:
#         return "Yellow"

# # Route for the homepage
# @app.route('/')
# def home():
#     return render_template('index.html')  # HTML file for uploading images

# # Route for processing the uploaded image
# @app.route('/detect', methods=['POST'])
# def detect():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded.'})
    
#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file.'})
    
#     # Save the uploaded image
#     image_path = os.path.join('uploads', file.filename)
#     os.makedirs('uploads', exist_ok=True)
#     file.save(image_path)
    
#     # Load and preprocess the image
#     image = Image.open(image_path)
#     image_tensor = transform(image).unsqueeze(0).to(device)
    
#     # Run inference
#     with torch.no_grad():
#         prediction = model(image_tensor)
    
#     # Process predictions
#     boxes = prediction[0]['boxes'].cpu().numpy()
#     labels = prediction[0]['labels'].cpu().numpy()
#     scores = prediction[0]['scores'].cpu().numpy()
    
#     traffic_light_class_label = 10  # Class label for traffic lights
#     threshold = 0.5
#     traffic_light_indices = labels == traffic_light_class_label
#     traffic_light_boxes = boxes[traffic_light_indices & (scores > threshold)]
#     traffic_light_scores = scores[traffic_light_indices & (scores > threshold)]
    
#     # Visualization
#     fig, ax = plt.subplots(1, figsize=(12, 9))
#     ax.imshow(image)
    
#     for box, score in zip(traffic_light_boxes, traffic_light_scores):
#         box = box.astype(int)
#         cropped_img = image.crop((box[0], box[1], box[2], box[3]))
#         light_color = detect_traffic_light_color(cropped_img)
        
#         box_color = {"Green": "green", "Red": "red", "Yellow": "yellow"}.get(light_color, "white")
#         ax.add_patch(plt.Rectangle(
#             (box[0], box[1]),
#             box[2] - box[0],
#             box[3] - box[1],
#             fill=False,
#             color=box_color,
#             linewidth=3
#         ))
#         ax.text(
#             box[0], box[1],
#             f'Traffic Light ({light_color}) ({score:.2f})',
#             fontsize=12,
#             color='black',
#             bbox=dict(facecolor=box_color, alpha=0.5)
#         )
    
#     output_path = os.path.join('outputs', f"result_{file.filename}")
#     os.makedirs('outputs', exist_ok=True)
#     plt.savefig(output_path)
#     plt.close()
    
#     return send_file(output_path, mimetype='image/png')

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, request, jsonify, render_template
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import cv2
# import io

# # Set the project name
# PROJECT_NAME = 'traffic_light_detection'

# # Initialize Flask app
# app = Flask(__name__)

# # Set the path to your saved model checkpoint
# MODEL_PATH = 'traffic_light_detection/models/fasterrcnn_resnet50_fpn.pth'  # Change this to your model's actual path

# # Initialize the Faster R-CNN model
# model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # Initialize with no weights
# model.load_state_dict(torch.load(MODEL_PATH))  # Load the saved model weights
# model.eval()  # Set the model to evaluation mode

# # Set the device for model inference (CUDA if available, else CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define the transform to convert the image into a tensor
# transform = transforms.Compose([transforms.ToTensor()])

# # Function to detect traffic light color
# def detect_traffic_light_color(cropped_img):
#     cropped_img = np.array(cropped_img)
#     hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)

#     # Color ranges for traffic lights
#     green_range = ((35, 50, 50), (85, 255, 255))
#     red_range_1 = ((0, 100, 100), (10, 255, 255))
#     red_range_2 = ((170, 100, 100), (180, 255, 255))
#     yellow_range = ((20, 100, 100), (30, 255, 255))

#     green_mask = cv2.inRange(hsv_img, green_range[0], green_range[1])
#     red_mask_1 = cv2.inRange(hsv_img, red_range_1[0], red_range_1[1])
#     red_mask_2 = cv2.inRange(hsv_img, red_range_2[0], red_range_2[1])
#     red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
#     yellow_mask = cv2.inRange(hsv_img, yellow_range[0], yellow_range[1])

#     green_percentage = np.sum(green_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
#     red_percentage = np.sum(red_mask) / (cropped_img.shape[0] * cropped_img.shape[1])
#     yellow_percentage = np.sum(yellow_mask) / (cropped_img.shape[0] * cropped_img.shape[1])

#     if green_percentage > red_percentage and green_percentage > yellow_percentage:
#         return "Green"
#     elif red_percentage > yellow_percentage:
#         return "Red"
#     else:
#         return "Yellow"

# @app.route('/')
# def serve_frontend():
#     return render_template('index.html')

# @app.route('/detect', methods=['POST'])
# def detect_traffic_lights():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     image = Image.open(io.BytesIO(file.read()))
#     image_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         prediction = model(image_tensor)

#     boxes = prediction[0]['boxes'].cpu().numpy()
#     labels = prediction[0]['labels'].cpu().numpy()
#     scores = prediction[0]['scores'].cpu().numpy()

#     threshold = 0.5
#     filtered_boxes = boxes[scores > threshold]
#     filtered_scores = scores[scores > threshold]

#     detection_results = []
#     for box, score in zip(filtered_boxes, filtered_scores):
#         cropped_img = image.crop((box[0], box[1], box[2], box[3]))
#         light_color = detect_traffic_light_color(cropped_img)
#         detection_results.append({
#             "box": box.tolist(),
#             "score": float(score),
#             "light_color": light_color
#         })

#     return jsonify(detection_results)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import io

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

# COCO class labels (You can modify this based on your custom dataset)
# 10 corresponds to "traffic light" in COCO dataset
COCO_CLASSES = {
    0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 
    5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 
    10: "traffic light", 11: "fire hydrant", 12: "stop sign", 13: "parking meter", 
    14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 
    20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 
    25: "orange", 26: "apple", 27: "banana", 28: "pineapple", 29: "grapes", 
    30: "watermelon", 31: "hot dog", 32: "pizza", 33: "donut", 34: "cake", 
    35: "chair", 36: "couch", 37: "potted plant", 38: "bed", 39: "dining table", 
    40: "toilet", 41: "tv", 42: "laptop", 43: "mouse", 44: "remote", 
    45: "keyboard", 46: "cell phone", 47: "microwave", 48: "oven", 
    49: "toaster", 50: "sink", 51: "refrigerator", 52: "book", 53: "clock", 
    54: "vase", 55: "scissors", 56: "teddy bear", 57: "hair drier", 
    58: "toothbrush"
}

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

@app.route('/')
def serve_frontend():
    return render_template('index.html')

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

    # Set the traffic light class label (10 in COCO dataset)
    traffic_light_class_label = 10

    # Filter out non-traffic light detections based on class label
    threshold = 0.5  # Minimum score to consider as valid detection
    traffic_light_boxes = boxes[scores > threshold]
    traffic_light_labels = labels[scores > threshold]
    traffic_light_scores = scores[scores > threshold]

    # Keep only traffic light class detections
    traffic_light_boxes = traffic_light_boxes[traffic_light_labels == traffic_light_class_label]
    traffic_light_scores = traffic_light_scores[traffic_light_labels == traffic_light_class_label]

    detection_results = []
    for box, score in zip(traffic_light_boxes, traffic_light_scores):
        cropped_img = image.crop((box[0], box[1], box[2], box[3]))  # Crop the region of interest
        light_color = detect_traffic_light_color(cropped_img)
        detection_results.append({
            "box": box.tolist(),
            "score": float(score),
            "light_color": light_color
        })

    return jsonify(detection_results)

if __name__ == '__main__':
    app.run(debug=True)

