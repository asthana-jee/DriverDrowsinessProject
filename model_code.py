import os
import pandas as pd
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def plot_img(arr):
    c = 0
    for i in arr:
        c += 1
        if c >= 5:
            break
        plt.figure(figsize=(2,2))
        plt.savefig(f'img_{c}.png')

def plot_loss(history, num_epochs):
    if num_epochs > len(history.history['loss']):
        raise ValueError("num_epochs should be less than or equal to the total number of epochs in the history.")

    start_epoch = len(history.history['loss']) - num_epochs
    end_epoch = len(history.history['loss'])

    plt.figure(figsize=(6, 4))
    plt.plot(range(start_epoch + 1, end_epoch + 1), history.history['loss'][start_epoch:], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss for Epochs {start_epoch + 1} - {end_epoch}')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')

def plot_training_history(history, from_=0):
    # Get training and validation loss from history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_validation_loss.png')

yawn_path = 'yawn-Dataset/train/0'
no_yawn_path = 'yawn-Dataset/train/1'
close_path = 'eyes/TrainingSet/TrainingSet/Closed'
open_path = 'eyes/TrainingSet/TrainingSet/Opened'

num_classes = {'Yawn': len(os.listdir(yawn_path)), 
               'No Yawn': len(os.listdir(no_yawn_path)), 
               'Closed': len(os.listdir(close_path)), 
               'Opened': len(os.listdir(open_path))}


# Loading Datasets

def load_images(yawn_path, no_yawn_path, target_shape=(32,32)):
    yawn_images = []
    no_yawn_images = []

    for filename in os.listdir(yawn_path):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            image_path = os.path.join(yawn_path, filename)
            img = Image.open(image_path)
            img = img.convert("RGB")  # Convert to RGB mode
            if target_shape is not None:
                img = img.resize(target_shape)  # Resize image
            yawn_images.append(np.array(img))
    
    for filename in os.listdir(no_yawn_path):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            image_path = os.path.join(no_yawn_path, filename)
            img = Image.open(image_path)
            img = img.convert("RGB")  # Convert to RGB mode
            if target_shape is not None:
                img = img.resize(target_shape)  # Resize image
            no_yawn_images.append(np.array(img))
    
    return np.array(yawn_images), np.array(no_yawn_images)

yawn_images, no_yawn_images = load_images(yawn_path, no_yawn_path)
closed_images, open_images = load_images(close_path, open_path)

all_yawn_y = np.full(len(yawn_images),'yawn')
all_no_yawn_y = np.full(len(no_yawn_images),'no_yawn')
all_close_y = np.full(len(closed_images),'close')
all_open_y = np.full(len(open_images),'open')

# Create labels for the images
labels = np.concatenate([all_yawn_y, all_no_yawn_y, all_close_y, all_open_y])
df = pd.DataFrame(labels, columns=['Labels'])
X = np.concatenate([yawn_images, no_yawn_images, closed_images, open_images])

# Normalize the images
X = X.astype('float32') / 255

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, df.values, test_size=0.2, random_state=seed_value)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed_value)

# Model Creation
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

def preprocess_img(image, label):
    image = tf.image.resize(image, (32, 32))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(preprocess_img)
val_dataset = val_dataset.map(preprocess_img)
test_dataset = test_dataset.map(preprocess_img)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# Training the Model
img_height, img_width = 32, 32
batch_size = 32
epochs = 15

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

# load data
train_generator = train_datagen.flow_from_directory('yawn-eye-Dataset/dataset_new/train',target_size=(img_height, img_width),batch_size=batch_size,class_mode='categorical')
valid_generator = validation_datagen.flow_from_directory('yawn-eye-Dataset/dataset_new/test',target_size=(img_height, img_width),batch_size=batch_size,class_mode='categorical')

# Model Architecture
model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Training the model
history = model.fit(train_generator, epochs=epochs, validation_data=valid_generator, callbacks=[early_stopping, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(valid_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Plot Training History
plot_training_history(history)

# Save the model
model.save('Drowsy_Driver_Detection.h5')

test_loss, test_acc = model.evaluate(valid_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")


