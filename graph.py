import matplotlib.pyplot as plt
import numpy as np
import random
# /255
# neurons = [10, 30, 50, 70, 90, 110, 150, 200, 300]
# accuracy = [0.0333, 0.1447, 0.7047, 0.9348, 0.9799, 0.9536, 0.9913, 0.9898, 0.9896]

# without /255
neurons = [128, 200, 300, 400, 500]
accuracy = [0.0607, 0.0480, 0.0602, 0.7134, 0.8023]

plt.plot(neurons, accuracy)

plt.xlabel("neurons allocated")
plt.ylabel("accuracy")




plt.show()
