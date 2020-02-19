Result

Network





Result
1500/1500 [==============================] - 0s 84us/sample - loss: 3.2210 - accuracy: 0.0900
Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(77, 68)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(30)
])