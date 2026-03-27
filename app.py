from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model(os.path.join(os.getcwd(), 'models', 'fire_detection_model.h5'))

def predict_image(img):
    """Process the image and predict if it contains fire."""
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    return 'Fire' if prediction[0][0] > 0.5 else 'Non-Fire'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Read and process the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    result = predict_image(img)
    
    # Render result page with prediction
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
