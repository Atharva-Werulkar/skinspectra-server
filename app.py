import cv2
import numpy as np
from flask import Flask, request, jsonify
import os

# Load your ML model (replace with your specific model loading code)
model = cv2.dnn_DetectionModel('path/to/your/model.pb', 'path/to/your/model.pbtxt')

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image'].read()
    image_array = cv2.imdecode(np.frombuffer(image, np.uint8), -1)

    # Preprocess the image (adjust based on your model's requirements)
    image_array = cv2.resize(image_array, (224, 224))  # Example resizing for a common model input size
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)  # Example color conversion
    image_array = image_array.astype(np.float32) / 255.0  # Example normalization
    image_array = np.expand_dims(image_array, axis=0)  # Example adding a batch dimension

    # Make predictions using your model
    class_ids, confidences, bboxes = model.detect(image_array, confThreshold=0.5)

    if len(class_ids) == 0:
        return jsonify({'prediction': 'No disease detected'})

    # Extract class labels from your model's configuration (replace as needed)
    with open('path/to/your/labels.txt', 'r') as f:
        labels = f.read().rstrip().split('\n')

    predicted_class = labels[int(class_ids[0][0])]
    predicted_confidence = confidences[0][0]

    return jsonify({'prediction': {'class': predicted_class, 'confidence': predicted_confidence}})

if __name__ == '__main__':
    app.run(debug=True)  # Bind to all interfaces for deployment
