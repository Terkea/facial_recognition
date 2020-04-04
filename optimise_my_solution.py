import tensorflow as tf
from tensorflow import keras
# tf.get_logger().setLevel('INFO')
from sklearn.model_selection import train_test_split

import os
import cv2
import shutil  
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from matplotlib import image
from tensorboard.plugins.hparams import api as hp

# memory fix
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


PATH = os.getcwd()
DATASET = os.path.join(PATH, "dataset")
CATEGORIES = [folder for folder in os.listdir(DATASET)]
INDEX_VALUES = [i for i in range(30)]


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

X_train, X_test, y_train, y_test = train_test_split(
    training_images, training_labels, test_size=0.3, random_state=42)


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([1, 100]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'


with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )



def train_test_model(hparams): 
    log_dir = "logs\\fit\\" \
    + "neurons-" + str(hparams[HP_NUM_UNITS]) + " " \
    + "dropout-" + str(hparams[HP_DROPOUT]) + " " \
    + str(hparams[HP_OPTIMIZER]) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " " + \
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(30, activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(np.array(X_train), np.array(y_train), epochs=10) # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(np.array(X_train), np.array(y_train)
    ,
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir),  # log metrics
        hp.KerasCallback(log_dir, hparams),  # log hparams
    ],)
    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1