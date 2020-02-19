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

# indexing the pictures by their label
def create_cvs():
    files = [file for file in os.listdir(DATADIR)]
    person = [file[5:7] for file in files]

    dataset = {'file': files,
            'person': person
            }
    df = DataFrame(dataset, columns=['file', 'person'])
    export_csv = df.to_csv('batch.csv', sep=',', encoding='utf-8', index=False)

def display_image(number):
    plt.figure()
    plt.imshow(train_images[number])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def display_more_images():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.show()


# loading the data for processing
df = pd.read_csv(BATCH)

train_images = [cv2.imread(DATADIR + f"/{image}", cv2.IMREAD_GRAYSCALE) for image in df['file']]
train_images = np.asarray(train_images)
train_images = train_images / 255.0

train_labels = df['person']

# building the neuronal network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(77, 68)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(30)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
