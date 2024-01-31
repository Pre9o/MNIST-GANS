import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras import backend as K

img_rows = 28
img_cols = 28
channels = 1

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
latent_dim = 100

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def build_generator(z_dim):
    model = Sequential()

    # Reshape input into 7x7x256 tensor via a fully connected layer
    model.add(Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Reshape((7, 7, 256)))

    # Transposed convolution layer, from 7x7x256 into 14x14x128 tensor
    model.add(Conv2DTranspose(128, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU())

    # Transposed convolution layer, from 14x14x128 to 14x14x64 tensor
    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 14x14x64 to 28x28x1 tensor
    model.add(Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', activation='tanh'))

    return model


def build_critic(img_shape):
    model = Sequential()

    # Convolutional layer, from 28x28x1 into 14x14x32 tensor
    model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=img_shape, padding='same'))
    
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    # Convolutional layer, from 14x14x32 into 7x7x64 tensor
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    
    # Flatten the tensor
    model.add(Flatten())
    model.add(Dense(1))

    return model


def build_gan(generator, critic):
    model = Sequential()

    # Combined Generator -> critic model
    model.add(generator)
    model.add(critic)

    return model

# Build and compile the critic
critic = build_critic(img_shape)
critic.compile(loss=wasserstein_loss,
                      optimizer=RMSprop(learning_rate=0.00005),
                      metrics=['accuracy'])

critic.treinable = False

# Build the Generator
generator = build_generator(latent_dim)

print('\n \n \n')

# Keep critic’s parameters constant for Generator training
critic.trainable = False

z = tf.keras.layers.Input(shape=(latent_dim,))
img = generator(z)

valid = critic(img)

# Build and compile GAN model with fixed critic to train the Generator
gan = tf.keras.models.Model(z, valid)
gan.compile(loss=wasserstein_loss, optimizer=RMSprop(learning_rate=0.00005))


def train(epochs, batch_size):
    # Load the MNIST dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = np.float32(X_train)
    X_train = (X_train / 255 - 0.5) * 2
    X_train = np.clip(X_train, -1, 1)

    real = -tf.ones(shape=(batch_size, 1))
    fake = tf.ones(shape=(batch_size, 1))

    d_loss = []
    g_loss = []

    for e in range(epochs + 1):
        for i in range(len(X_train) // batch_size):
            for _ in range(n_critic):

                # Train Discriminator weights
                critic.trainable = True

                # Real samples
                X_batch = X_train[i * batch_size : (i + 1) * batch_size]
                d_loss_real = critic.train_on_batch(x=X_batch, y=real)

                # Fake Samples
                z = tf.random.normal(
                    shape=(batch_size, latent_dim), mean=0, stddev=1
                )
                X_fake = generator.predict_on_batch(z)
                d_loss_fake = critic.train_on_batch(x=X_fake, y=fake)

                # Discriminator loss
                d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])

                # Clip critic weights
                for l in critic.layers:
                    weights = l.get_weights()
                    weights = [
                        np.clip(w, -clip_value, clip_value) for w in weights
                    ]
                    l.set_weights(weights)

            # Train Generator weights
            critic.trainable = False
            g_loss_batch = gan.train_on_batch(x=z, y=real)

            print(f"epoch = {e + 1}/{epochs}, batch = {i + 1}/{len(X_train) // batch_size}, d_loss = {d_loss_batch}, g_loss = {g_loss_batch}")

        d_loss.append(d_loss_batch)
        g_loss.append(g_loss_batch)
        print(f"epoch = {e + 1}/{epochs}, d_loss = {d_loss[-1]}, g_loss = {g_loss[-1]}")

        if e % 2 == 0:
            samples = 10

            z = tf.random.normal(shape=(samples, latent_dim), mean=0, stddev=1)
            x_fake = generator.predict_on_batch(z)

            fig = plt.figure(figsize=(4, 3))
            for k in range(samples):
                plt.subplot(2, 5, k + 1)
                plt.imshow(x_fake[k].reshape(28, 28), cmap="gray")
                plt.xticks([])
                plt.yticks([])

            plt.tight_layout()
            plt.show()
                        
            
# Set hyperparameters
epochs = 100
batch_size = 64
n_critic = 5
clip_value = 0.01

# Train the DCGAN for the specified number of iterations
train(epochs, batch_size)