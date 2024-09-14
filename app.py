from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import tensorflow as tf
from PIL import Image
import cv2


# Loading trained model
model = load_model('my_model1.keras')  
app = Flask(__name__)


# Home route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')
    

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    
    if 'image' not in request.files:
        return render_template('index.html', error='No image uploaded!')

    file = request.files['image']
    
    
    filepath = os.path.join('static', file.filename)
    file.save(filepath)
    print(f'filepath:{filepath}')
    print(f'file:{file}')
    # Process the image to be fed to the model for prediction
    image = cv2.imread(filepath)
    test_image = cv2.resize(image, (256 ,256))
    im=tf.constant(test_image, dtype=tf.float32 )  # Resizing the image to make it compatible with model
    im=tf.expand_dims(im, axis=0)
    # Predict emotion
    predictions = model.predict(im)
    emotion_labels = ['Angry', 'Happy', 'Sad']  # Emotion labels
    predicted_emotion = emotion_labels[np.argmax(predictions)]

    return render_template('result.html', emotion=predicted_emotion, image_file=filepath)

if __name__ == '__main__':
    app.run(debug=True)
