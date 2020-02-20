from __future__ import absolute_import, division, print_function, unicode_literals

# # TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import cv2
import shutil  
from PIL import Image
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

def normalize_dataset():
    for file in os.listdir("dataset"):
        try:
            os.mkdir(f"dataset/{file[5:7]}")
            
        except FileExistsError:
            print("Directory ", file[5:7], " already exists")

        shutil.move(os.getcwd() + f"/dataset/{file}", os.getcwd() + f"/dataset/{file[5:7]}/{file}")

PATH = os.getcwd()
DATASET = os.path.join(PATH, "dataset")
CATEGORIES = [folder for folder in os.listdir(DATASET)]
INDEX_VALUES = [i for i in range(30)]

# normalize_dataset()

def create_training_data():
    _images = []
    _labels = []

    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        new_path = os.path.join(DATASET, category)
        for img in os.listdir(new_path):
            img_array = cv2.imread(os.path.join(new_path, img), cv2.IMREAD_GRAYSCALE)
            _images.append(img_array)
            _labels.append(class_num)
    return (_images, _labels)

(training_images, training_labels) = create_training_data()

training_images = np.array(training_images)
training_images = training_images / 255.0

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(77, 68)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(30)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def train_model(model):
    checkpoint_path = "model/model.ckpt"
    checkpoint_dir = os.getcwd()

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    # Train the model with the new callback
    model.fit(x=np.array(training_images), 
            y=np.array(training_labels),  
            epochs=10,
            validation_data=(np.array(training_images),np.array(training_labels)),
            callbacks=[cp_callback])  # Pass callback to training

train_model(create_model())

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()