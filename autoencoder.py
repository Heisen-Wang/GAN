# a sample code for encode and decode MNIST
import numpy as np
np.random.seed(1337)

from keras.models import Model
from keras.datasets import mnist
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
## show the image
plt.figure(1)
for i in range(9):
    plt.subplot(331+i)
    plt.imshow(x_test[i, :, :])
plt.savefig('./figure/origin')

## data preprocessing
x_train = x_train.astype(np.float32)/255. - 0.5
x_test = x_test.astype(np.float32)/255. - 0.5
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_test.shape)

# build the model
encoding_dim = 2
input_img = Input(shape=(x_train.shape[1],))
## encode layers
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation = 'relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_out = Dense(encoding_dim, activation='relu')(encoded)

## decode layers
decoded = Dense(10, activation='relu')(encoder_out)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
output_img = Dense(x_train.shape[1], activation='tanh')(decoded)

auto_encoder = Model(input=input_img, output=output_img)
encoder = Model(input=input_img, output=encoder_out)

# compile
auto_encoder.compile(optimizer='adam', loss='mse')

# train
auto_encoder.fit(x_train, x_train, batch_size=256, epochs=40, shuffle=True)

# test
encoder_img = auto_encoder.predict(x_test)
img = encoder_img.reshape((encoder_img.shape[0], 28, 28))
plt.figure(2)
for i in range(9):
    plt.subplot(331+i)
    plt.imshow(img[i, :, :])
plt.savefig('./figure/result')