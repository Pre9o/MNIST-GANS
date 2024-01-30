import matplotlib.pyplot as plt
import numpy as np

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
z_dim = 100

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def build_generator(z_dim):
    model = Sequential()

    # Reshape input into 7x7x256 tensor via a fully connected layer
    model.add(Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(Reshape((7, 7, 256)))

    # Transposed convolution layer, from 7x7x256 into 14x14x128 tensor
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 14x14x128 to 14x14x64 tensor
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Transposed convolution layer, from 14x14x64 to 28x28x1 tensor
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))

    return model


def build_critic(img_shape):
    model = Sequential()

    # Convolutional layer, from 28x28x1 into 14x14x32 tensor
    model.add(
        Conv2D(32,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Convolutional layer, from 14x14x32 into 7x7x64 tensor
    model.add(
        Conv2D(64,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Convolutional layer, from 7x7x64 tensor into 3x3x128 tensor
    model.add(
        Conv2D(128,
               kernel_size=3,
               strides=2,
               input_shape=img_shape,
               padding='same'))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Dropout
    model.add(Dropout(0.6))

    # Output layer with sigmoid activation
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))

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
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

# Build the Generator
generator = build_generator(z_dim)

print('\n \n \n')

# Keep critic’s parameters constant for Generator training
critic.trainable = False

# Build and compile GAN model with fixed critic to train the Generator
gan = build_gan(generator, critic)
gan.compile(loss=wasserstein_loss, optimizer=RMSprop())


def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

    plt.show()


losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):
    # Load the MNIST dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)

    # Labels for real images: all ones
    real = np.ones((batch_size, 1))

    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  Train the critic
        # -------------------------

        for _ in range(n_critic):

            critic.trainable = True
            
            # Get a random batch of real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            
            # Generate a batch of fake images
            z = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = generator.predict(z)
            
            # Train Critic
            c_loss_real = critic.train_on_batch(imgs, real)
            c_loss_fake = critic.train_on_batch(gen_imgs, fake)
            
            # Total critic loss
            c_loss = 0.5 * np.add(c_loss_fake, c_loss_real)
            
            # Clip critic weights to satisfy Lipschitz condition
            for l in critic.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                l.set_weights(weights)
                
        critic.trainable = False
        
        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)
        
        # Train Generator
        g_loss = gan.train_on_batch(z, real)
        
        print(
            "%d [C loss: %f] [G loss: %f]"
            % (iteration, 1 - c_loss[0], 1 - g_loss)
        )
        
        if (iteration + 1) % sample_interval == 0:
            # Save losses and accuracies so they can be plotted after training
            losses.append((1 - c_loss[0], 1 - g_loss))
            accuracies.append(100 * c_loss[1])
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print(
                "%d [C loss: %f] [G loss: %f]"
                % (iteration + 1, 1 - c_loss[0], 1 - g_loss)
            )

            # Output a sample of generated image
            sample_images(generator)
                        
            
# Set hyperparameters
iterations = 20000
batch_size = 128
sample_interval = 100
n_critic = 5
clip_value = 0.01

# Train the DCGAN for the specified number of iterations
train(iterations, batch_size, sample_interval)