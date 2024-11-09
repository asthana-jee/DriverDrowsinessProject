import os
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.models import load_model
# Function to load and preprocess a new image
def load_single_image_from_link(image_link, target_shape=(32, 32)):
    response = requests.get(image_link)
    image = Image.open(BytesIO(response.content))
    image = image.convert("RGB")
    image = image.resize(target_shape)
    return np.array(image) / 255.0

# List of URLs to new images of people driving
new_image_urls = [
    'https://www.bornbee.com/wp-content/uploads/2024/07/drowsy-driver-picture-03.jpg',
    'https://www.bornbee.com/wp-content/uploads/2024/07/normal-driver-image-01.jpg',
    'https://www.bornbee.com/wp-content/uploads/2024/07/drowsy-driver-picture-01.jpg'
]
model = load_model("Drowsy_Driver_Detection.h5")
# Load and preprocess new images
new_images = np.array([load_single_image_from_link(url) for url in new_image_urls])

# Predict drowsiness
predictions = model.predict(new_images)

# Display results
for i, (image_url, prediction) in enumerate(zip(new_image_urls, predictions)):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.title('Drowsy' if np.argmax(prediction) in [0, 2] else 'Not Drowsy')
    plt.axis('off')
    plt.show()

