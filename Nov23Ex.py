# %%
from tensorflow import keras
from keras import layers
latent_dim = 2
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2,
padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x) 
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x) 
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
# %%
import tensorflow as tf
class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size)) 
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
# %%
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs) 
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
# %%
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs) 
        self.encoder = encoder 
        self.decoder = decoder 
        self.sampler = Sampler() 
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss") 
        # weighted mean class 
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss") 
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data) 
            z = self.sampler(z_mean, z_log_var) 
            reconstruction = decoder(z) 
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) 
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
            grads = tape.gradient(total_loss, self.trainable_weights) 
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) 
            self.total_loss_tracker.update_state(total_loss) 
            self.reconstruction_loss_tracker.update_state(reconstruction_loss) 
            self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(), 
            "reconstruction_loss": self.reconstruction_loss_tracker.result(), 
            "kl_loss": self.kl_loss_tracker.result(),
            }
# %%
import numpy as np
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data() 
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255 
vae = VAE(encoder, decoder)
# We don't need to pass the loss argument since it is already part of the train_step()
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True) 
vae.fit(mnist_digits, epochs=30, batch_size=128)
# %%
import matplotlib.pyplot as plt
n = 30 # we will generate 30x30 images of digits 
digit_size = 28 # each image is of size 28x28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-1, 1, n)
grid_y = np.linspace(-1, 1, n)[::-1]
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = vae.decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size) 
        figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size] = digit
plt.figure(figsize=(15, 15))
start_range = digit_size // 2
end_range = n * digit_size + start_range
pixel_range = np.arange(start_range, end_range, digit_size) 
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x) 
plt.yticks(pixel_range, sample_range_y) 
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.axis("off")
plt.imshow(figure, cmap="Greys_r")