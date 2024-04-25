from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os

app = Flask(__name__)

# Define the path to the h5 file
h5_file_path = os.path.join(os.path.dirname(__file__), '..', 'beef_pork_horse_classifier.h5')

# Load the model
model = tf.keras.models.load_model(h5_file_path)

@app.route('/', methods=['GET'])
def hello_world():
    return jsonify({'message': 'Hello, World!'})

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({'models': 'MobileNetV3Large', 'framework': 'TensorFlow', 'task': 'Image Classification for Beef, Pork, and Horse', 'accuracy': '97.43%', 'input': 'URL', 'output': 'Predicted class and probabilities', 'model_url': 'https://www.kaggle.com/code/hafidhmuhammadakbar/mobilenetv3large-fix'})

# post request from url form-data
# @app.route('/models', methods=['POST'])
# def predict():
#     # Check if the request contains a URL
#     if 'url' not in request.form:
#         return jsonify({'error': 'No URL provided'})

#     url = request.form['url']

#     # Download the image from the URL
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         image = Image.open(BytesIO(response.content))
#     except Exception as e:
#         return jsonify({'error': f'Failed to download image from URL: {str(e)}'})

#     # Resize image
#     image = image.resize((224, 224))

#     # Preprocess image
#     image = np.expand_dims(image, axis=0)

#     # Make prediction
#     predictions = model.predict(image)
#     predicted_label = np.argmax(predictions, axis=1)[0]
#     probabilities = tf.reduce_max(predictions, axis=1) * 100

#     class_names = ['Horse', 'Meat', 'Pork']
#     predicted_class = class_names[predicted_label]
#     probabilities_class = '%.2f' % probabilities.numpy()[0]

#     return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities_class})

# post request from url json
@app.route('/models', methods=['POST'])
def predict():
    # Check if the request contains JSON data
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'})

    # Get JSON data from request
    data = request.json

    # Check if the JSON data contains a URL
    if 'url' not in data:
        return jsonify({'error': 'No URL provided'})

    url = data['url']

    # Download the image from the URL
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        return jsonify({'error': f'Failed to download image from URL: {str(e)}'})

    # Resize image
    image = image.resize((224, 224))

    # Preprocess image
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions, axis=1)[0]
    probabilities = tf.reduce_max(predictions, axis=1) * 100

    class_names = ['Horse', 'Meat', 'Pork']
    predicted_class = class_names[predicted_label]
    probabilities_class = '%.2f' % probabilities.numpy()[0]

    return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities_class})