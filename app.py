import cv2
import numpy as np
from flask import Flask, request, jsonify
import joblib
import os

# Load your ML model (replace with your specific model loading code)
model = joblib.load('model.h5')

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
    class_ids, confidences = model.detect(image_array, confThreshold=0.5)

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


# from flask import Flask, request, jsonify, send_file
# import tensorflow as tf
# import numpy as np
# import os
# import PIL
# from PIL import Image

# app = Flask(__name__)

# # Load the saved model
# model = tf.keras.models.load_model('path/to/saved_model')

# @app.route('/preprocess', methods=['POST'])
# def preprocess():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     img_file = request.files['image']
#     if img_file.filename == '':
#         return jsonify({'error': 'No image provided'}), 400

#     img = Image.open(img_file)
#     img = img.resize((180, 180))
#     img_array = np.array(img) / 255.0

#     # Preprocess the image for the model
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     # Make a prediction using the model
#     predictions = model.predict(img_array)

#     # Add the prediction to the response
#     response = {'image_array': img_array.tolist(), 'predictions': predictions.tolist()}

#     return jsonify(response)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     img_file = request.files['image']
#     if img_file.filename == '':
#         return jsonify({'error': 'No image provided'}), 400

#     # Preprocess the image and make a prediction as before
#     img = Image.open(img_file)
#     img = img.resize((180, 180))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     predictions = model.predict(img_array)

#     # Send the prediction back to the user as an image
#     image = Image.fromarray((predictions[0] * 255).astype(np.uint8))
#     image.save('prediction.jpg')
#     return send_file('prediction.jpg', mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(debug=True)