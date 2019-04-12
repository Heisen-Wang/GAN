import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
plt.figure()
plt.scatter(X,Y)
plt.show()
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

model = Sequential()
model.add(Dense(output_dim = 1, input_dim = 1))

model.compile(loss='mse', optimizer='sgd')
