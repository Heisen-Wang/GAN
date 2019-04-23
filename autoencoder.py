# a sample code for encode and decode MNIST
import numpy as np
np.random.seed(1337)

from keras.models import Model
from keras.datasets import mnist
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# show the image
# fig = plt.figure()
# plt.imshow(x_train[1, :, :])
# plt.show()

# data preprocessing
x_train = x_train.astype(np.float32)/255. - 0.5
x_test = x_test.astype(np.float32)/255. - 0.5
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_test.shape)

# build the model