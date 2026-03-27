from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the previously trained model
model = load_model('fire_detection_model.h5')

def predict_image(img_path):
    # Load and preprocess the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unable to load at path: {img_path}")
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict class
    prediction = model.predict(img)
    return 'Fire' if prediction[0][0] > 0.5 else 'Non-Fire'

# Test the function with an example image
test_image_path = r"C:\Users\Lenovo\Desktop\Fire Detection\Fire Detection\images.jpg" # Replace with an actual test image path
result = predict_image(test_image_path)
print(f'The image is classified as: {result}')
