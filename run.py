from __future__ import absolute_import, division, print_function, unicode_literals

# # TensorFlow and tf.keras
# import tensorflow as tf
# from tensorflow import keras

# Helper libraries
import os
import cv2
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

DATADIR = "dataset"

# indexing the pictures by their label
def create_cvs():
    DIRDATA = 'dataset'
    files = [file for file in os.listdir(DATADIR)]
    person = [file[4:7] for file in files]

    dataset = {'file': files,
            'person': person
            }
    df = DataFrame(dataset, columns=['file', 'person'])
    export_csv = df.to_csv('labeled_data.csv', sep=',', encoding='utf-8', index=False)
