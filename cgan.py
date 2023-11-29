import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# reshaping the inputs
X_train = X_train.reshape(60000, 28*28)
# normalizing the inputs (-1, 1)
X_train = (X_train.astype('float32') / 255 - 0.5) * 2

# latent space dimension
latent_dim = 100

# imagem dimension 28x28
img_dim = 784

seed_value = 42
initializer = tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02, seed=seed_value
)


# Generator network
generator = tf.keras.models.Sequential()

# Input layer and hidden layer 1
generator.add(
    tf.keras.layers.Dense(
        128, input_shape=(latent_dim,), kernel_initializer=initializer
    )
)
generator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
generator.add(tf.keras.layers.BatchNormalization(momentum=0.8))

# Hidden layer 2
generator.add(tf.keras.layers.Dense(256))
generator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
generator.add(tf.keras.layers.BatchNormalization(momentum=0.8))

# Hidden layer 3
generator.add(tf.keras.layers.Dense(512))
generator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
generator.add(tf.keras.layers.BatchNormalization(momentum=0.8))

# Output layer
generator.add(tf.keras.layers.Dense(img_dim, activation="tanh"))


# Embedding condition in input layer
num_classes = 10

# Create label embeddings
label = tf.keras.layers.Input(shape=(1,), dtype='int32')
label_embedding = tf.keras.layers.Embedding(num_classes, latent_dim)(label)
label_embedding = tf.keras.layers.Flatten()(label_embedding)

# latent space
z = tf.keras.layers.Input(shape=(latent_dim,))

# Merge inputs (z x label)
input_generator = tf.keras.layers.multiply([z, label_embedding])

# Output image
img = generator(input_generator)

# Generator with condition input
generator = tf.keras.models.Model([z, label], img)


# Discriminator network
discriminator = tf.keras.models.Sequential()

# Input layer and hidden layer 1
discriminator.add(
    tf.keras.layers.Dense(
        128, input_shape=(img_dim,), kernel_initializer=initializer
    )
)
discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))

# Hidden layer 2
discriminator.add(tf.keras.layers.Dense(256))
discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))

# Hidden layer 3
discriminator.add(tf.keras.layers.Dense(512))
discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))

# Output layer
discriminator.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Embedding condition in input layer

# Create label embeddings
label_d = tf.keras.layers.Input(shape=(1,), dtype='int32')
label_embedding_d = tf.keras.layers.Embedding(num_classes, img_dim)(label_d)
label_embedding_d = tf.keras.layers.Flatten()(label_embedding_d)

# imagem dimension 28x28
img_d = tf.keras.layers.Input(shape=(img_dim,))

# Merge inputs (img x label)
input_discriminator = tf.keras.layers.multiply([img_d, label_embedding_d])

# Output image
validity = discriminator(input_discriminator)

# Discriminator with condition input
discriminator = tf.keras.models.Model([img_d, label_d], validity)

# Optimizer
optimizer = tf.keras.optimizers.legacy.Adam()


discriminator.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)


discriminator.trainable = False

validity = discriminator([generator([z, label]), label])

d_g = tf.keras.models.Model([z, label], validity)

d_g.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)

def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
    samples = 10

    # Sample random noise
    z = tf.random.normal(shape=(samples, latent_dim), mean=0, stddev=1)
    labels = np.arange(0, 10).reshape(-1, 1)

    # Generate images from random noise
    gen_imgs = generator.predict([z, labels])

    fig = plt.figure(figsize=(4, 3))
    for k in range(samples):
        plt.subplot(2, 5, k + 1)
        plt.imshow(gen_imgs[k].reshape(28, 28), cmap="gray")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()
    
    
epochs = 200
batch_size = 64
smooth = 0.1

real = tf.ones(shape=(batch_size, 1))
fake = tf.zeros(shape=(batch_size, 1))

d_loss = []
d_g_loss = []

for e in range(epochs + 1):
    for i in range(len(X_train) // batch_size):

        # Train Discriminator weights
        discriminator.trainable = True

        # Real samples
        X_batch = X_train[i * batch_size : (i + 1) * batch_size]
        real_labels = y_train[i * batch_size : (i + 1) * batch_size].reshape(
            -1, 1
        )

        d_loss_real = discriminator.train_on_batch(
            x=[X_batch, real_labels], y=real * (1 - smooth)
        )

        # Fake Samples
        z = tf.random.normal(shape=(batch_size, latent_dim), mean=0, stddev=1)
        random_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
        X_fake = generator.predict_on_batch([z, random_labels])

        d_loss_fake = discriminator.train_on_batch(
            x=[X_fake, random_labels], y=fake
        )

        # Discriminator loss
        d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        # Train Generator weights
        discriminator.trainable = False

        z = tf.random.normal(shape=(batch_size, latent_dim), mean=0, stddev=1)
        random_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
        d_g_loss_batch = d_g.train_on_batch(x=[z, random_labels], y=real)

        print(
            "epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f"
            % (
                e + 1,
                epochs,
                i,
                len(X_train) // batch_size,
                d_loss_batch,
                d_g_loss_batch[0],
            ),
            100 * " ",
            end="\r",
        )

    d_loss.append(d_loss_batch)
    d_g_loss.append(d_g_loss_batch[0])

    print(
        "epoch = %d/%d, d_loss=%.3f, g_loss=%.3f"
        % (e + 1, epochs, d_loss[-1], d_g_loss[-1]),
        100 * " ",
    )

    if e % 5 == 0:
        sample_images(generator)