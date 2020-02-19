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

# loading the data for processing
df = pd.read_csv(BATCH)

train_images = [cv2.imread(DATADIR + f"/{image}", cv2.IMREAD_GRAYSCALE) for image in df['file']]
train_images = np.asarray(train_images)

train_labels = df['person']


