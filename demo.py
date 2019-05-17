from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np


class GAN():
    def __init__(self):
        self.x_dim = 1
        self.noise_dim = 10
        self.channels = 1
        self.y_shape = (self.x_dim,)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # generator takes x as input
        x = Input(shape=(self.x_dim,))
        x = Input(shape=(self.x_dim + self.noise_dim,))
        y = self.generator(x)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(y)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(x, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        initilization_method = 'he_normal'  # 'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'

        x_shape = (1,)
        x_shape = (1 + self.noise_dim,)

        model = Sequential()
        model.add(Dense(20, input_shape=x_shape, kernel_initializer=initilization_method))
        model.add(Dense(20, activation='relu', kernel_initializer=initilization_method))
        # model.add(Dense(20, activation='relu', kernel_initializer=initilization_method))
        # model.add(Dense(20, activation='relu', kernel_initializer=initilization_method))
        model.add(Dense(32, activation='linear', kernel_initializer=initilization_method))

        model.add(Dense(20, activation='relu', kernel_initializer=initilization_method))
        model.add(Dense(1, activation='linear', kernel_initializer=initilization_method))

        model.summary()

        x = Input(shape=x_shape)
        y = model(x)

        return Model(x, y)

    def build_discriminator(self):
        initilization_method = 'he_normal'  # 'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'

        dropout = 0.0
        y_shape = (self.x_dim,)

        model = Sequential()
        model.add(Dense(20, input_shape=y_shape, kernel_initializer=initilization_method))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(20, activation='relu', kernel_initializer=initilization_method))
        # model.add(Dense(20, activation='relu', kernel_initializer=initilization_method))
        model.add(Dense(80, activation='relu', kernel_initializer=initilization_method))
        model.add(Dense(80, activation='relu', kernel_initializer=initilization_method))
        model.add(Dense(80, activation='relu', kernel_initializer=initilization_method))
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initilization_method))
        model.summary()

        y = Input(shape=y_shape)
        validity = model(y)

        return Model(y, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        training_size = 100000;
        x_train = (np.random.randint(2, size=training_size) * 2 - 1).reshape(training_size, 1)
        # x_train = np.zeros(training_size).reshape(training_size,1)

        mu, sigma = 0, 0.5  # mean and standard deviation
        noise = np.random.normal(mu, sigma, training_size).reshape(training_size, 1)
        y_train = x_train + noise
        self.show_hist(x_train, y_train)

        # random_noise
        # # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            y = y_train[idx]

            x = x_train[idx]
            mu, sigma = 0, 1  # mean and standard deviation
            input_noise = np.random.normal(mu, sigma, half_batch * self.noise_dim).reshape(half_batch, self.noise_dim)

            # Generate a half batch of new ys
            input_x = np.concatenate((x, input_noise), 1)
            gen_y = self.generator.predict(input_x)

            # Train the discriminator
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(y, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_y, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            idx = np.random.randint(0, x_train.shape[0], batch_size)
            x = x_train[idx]

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            self.discriminator.trainable = False

            mu, sigma = 0, 1  # mean and standard deviation
            input_noise = np.random.normal(mu, sigma, batch_size * self.noise_dim).reshape(batch_size, self.noise_dim)
            input_x = np.concatenate((x, input_noise), 1)

            g_loss = self.combined.train_on_batch(input_x, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                test_size = 10000
                x_test = (np.random.randint(2, size=test_size) * 2 - 1).reshape(test_size, 1)
                # x_test = np.zeros(test_size).reshape(test_size,1)

                mu, sigma = 0, 1  # mean and standard deviation
                input_noise = np.random.normal(
                    mu, sigma, test_size * self.noise_dim).reshape(test_size, self.noise_dim)
                input_x = np.concatenate((x_test, input_noise), 1)
                gen_ys = self.generator.predict(input_x)
                self.show_hist(x_test, gen_ys)

    def show_hist(self, x, y):
        # =============================================================================
        #         r, c = 5, 5
        #         noise = np.random.normal(0, 1, (r * c, 100))
        # =============================================================================

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5
        import seaborn as sns
        idx = x > 0
        sns.distplot(y[idx], color='red', kde=False)
        sns.distplot(y[~idx], color='blue', kde=False)
        plt.show()


if __name__ == '__main__':
    plt.close('all')
    gan = GAN()
    gan.train(epochs=10000, batch_size=100, save_interval=100)

    # #test for scalibility
    # test_size = 10000
    # x_test = (np.random.randint(2, size=test_size) * 8 - 4).reshape(test_size,1)
    # mu, sigma = 0, 1 # mean and standard devia
    # input_x = np.concatenate((x_test, input_noise),1)
    # ys= gan.generator.predict(input_x)
    # gan.show_hist(x_test, ys)

    # test_size = 10000
    # x_test = (np.random.randint(2, size=test_size) * 2 - 1).reshape(test_size,1)
    # mu, sigma = 0, 10 # mean and standard devia
    # input_x = np.concatenate((x_test, input_noise),1)
    # ys= gan.generator.predict(input_x)
    # gan.show_hist(x_test, ys)

