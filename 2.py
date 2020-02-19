from __future__ import absolute_import, division, print_function, unicode_literals

# # TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import cv2
from PIL import Image
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

PATH = os.getcwd()
DATADIR = PATH + "/dataset"
BATCH = PATH + "/batch.csv"

# loading the data for processing
df = pd.read_csv(BATCH)

train_images = [cv2.imread(DATADIR + f"/{image}", cv2.IMREAD_GRAYSCALE) for image in df['file']]
train_images = np.asarray(train_images)
train_images = train_images

train_labels = df['person']

class_names = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 37, 38, 39]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(77, 68)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(30)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)