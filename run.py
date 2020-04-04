import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel('INFO')
from sklearn.model_selection import train_test_split


import os
import cv2
import shutil  
import numpy as np
import random
import datetime

PATH = os.getcwd()
DATASET = os.path.join(PATH, "dataset")
CATEGORIES = [folder for folder in os.listdir(DATASET)]
INDEX_VALUES = [i for i in range(30)]

def normalize_dataset():
    for file in os.listdir("dataset"):
        try:
            os.mkdir(f"dataset/{file[5:7]}")
            
        except FileExistsError:
            print("Directory ", file[5:7], " already exists")

        shutil.move(os.getcwd() + f"/dataset/{file}", os.getcwd() + f"/dataset/{file[5:7]}/{file}")

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


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(77, 68)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(30)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def train_model(model, train_data, train_labels):
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x=np.array(training_images), 
            y=np.array(training_labels),  
            epochs=10,
            validation_data=(np.array(train_data), np.array(train_labels)),
            callbacks=[tensorboard_callback])

    return model

if __name__ == "__main__":
    (training_images, training_labels) = create_training_data()
    training_images = np.array(training_images)
    training_images = training_images / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        training_images, training_labels, test_size=0.3, random_state=42)

    print("create and train the model \n")
    train_model(create_model(), X_train, y_train)