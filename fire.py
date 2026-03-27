import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split

# Define paths
fire = r"C:\Users\Lenovo\Desktop\Fire Detection\Fire Detection\fire_dataset\fire_images"
non_fire = r"C:\Users\Lenovo\Desktop\Fire Detection\Fire Detection\fire_dataset\non_fire_images"

# Load and preprocess data
def load_images_from_folder(folder, label, img_size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return images, labels

fire_images, fire_labels = load_images_from_folder(fire, 1)
non_fire_images, non_fire_labels = load_images_from_folder(non_fire, 0)

all_images = np.array(fire_images + non_fire_images)
all_labels = np.array(fire_labels + non_fire_labels)
all_images = all_images / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42
)

# Define the model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# Save the trained model
model.save('fire_detection_model.h5')
