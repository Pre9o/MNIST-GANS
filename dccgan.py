import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# load dataset
(X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 127.5 - 1.0  # Normalize to the range [-1, 1]

# Input shape
img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)

# Latent space dimension
latent_dim = 100

# Number of classes
num_classes = 10

# Build the Generator
def build_generator():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(128 * 7 * 7, input_dim=latent_dim))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Reshape((7, 7, 128)))

    model.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2DTranspose(channels, kernel_size=3, strides=2, padding="same", activation="tanh"))

    noise = tf.keras.layers.Input(shape=(latent_dim,))
    label = tf.keras.layers.Input(shape=(1,), dtype='int32')

    label_embedding = tf.keras.layers.Embedding(num_classes, latent_dim)(label)
    flat_embedding = tf.keras.layers.Flatten()(label_embedding)

    model_input = tf.keras.layers.multiply([noise, flat_embedding])
    img = model(model_input)

    return tf.keras.models.Model(inputs=[noise, label], outputs=img)


# Build the Discriminator
def build_discriminator():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=img_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    img = tf.keras.layers.Input(shape=img_shape)
    label = tf.keras.layers.Input(shape=(1,), dtype='int32')

    label_embedding = tf.keras.layers.Embedding(num_classes, np.prod(img_shape))(label)
    flat_embedding = tf.keras.layers.Flatten()(label_embedding)
    flat_img = tf.keras.layers.Flatten()(img)

    model_input = tf.keras.layers.multiply([flat_img, flat_embedding])
    validity = model(model_input)

    return tf.keras.models.Model(inputs=[img, label], outputs=validity)


# Compile the Discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# Compile the Generator
generator = build_generator()
z = tf.keras.layers.Input(shape=(latent_dim,))
label = tf.keras.layers.Input(shape=(1,))
img = generator([z, label])

discriminator.trainable = False

validity = discriminator([img, label])

combined = tf.keras.models.Model([z, label], validity)
combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

def sample_images(generator, epoch, rows=4, columns=4):
    r, c = rows, columns
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    sampled_labels = np.arange(0, 10).reshape(-1, 1)
    gen_imgs = generator.predict([noise, sampled_labels])

    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.show()

# Training
epochs = 200
batch_size = 64

half_batch = int(batch_size / 2)

for epoch in range(epochs + 1):
    for _ in range(len(X_train) // batch_size):
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs, labels = X_train[idx], y_train[idx]

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        gen_imgs = generator.predict([noise, labels])

        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

        valid = np.ones((batch_size, 1))

        g_loss = combined.train_on_batch([noise, labels], valid)

    print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
    if epoch % 10 == 0:
        sample_images(generator, epoch)
