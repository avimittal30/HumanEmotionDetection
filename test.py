from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
# from PIL import Image
# import cv2

# Load your pre-trained model
model = load_model('my_model.keras')  # Replace with your model path

filepath = os.path.join('static', file.filename)
file.save(filepath)

# Process the image
image = Image.open(file)
image = image.convert('RGB')
image = image.resize((48, 48))  # Resize the image for the model
image = img_to_array(image)
image = np.expand_dims(image, axis=0) / 255.0  # Normalize the image

