import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os

# 1. Configuración de hiperparámetros
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 40 # Le damos un poco más de épocas al VAE para que organice sus distribuciones
DATA_DIR = "./dataset/transistor/train"
LATENT_DIM = 128 # Aumentamos el espacio latente porque nuestras imágenes son más grandes y complejas

print(f"Cargando imágenes de: {DATA_DIR}...")

# 2. Cargar y preprocesar los datos
dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels=None, # Solo queremos las imágenes, sin etiquetas
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Se necesita los datos de entrada exactamente entre 0 y 1
normalization_layer = layers.Rescaling(1./255)
# Se mapea 'x' (la imagen de entrada), el VAE se encarga del resto
dataset = dataset.map(lambda x: normalization_layer(x))

# 3. Capa de Muestreo 
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

# 4. Construir el ENCODER Convolucional
encoder_input = layers.Input(shape=(128, 128, 3), name="encoder_input")
x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_input) # Baja a 64x64
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)             # Baja a 32x32
x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)            # Baja a 16x16
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)

# Salidas del codificador: media y varianza
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# 5. Construir el DECODER Convolucional
decoder_input = layers.Input(shape=(LATENT_DIM,), name="decoder_input")
x = layers.Dense(16 * 16 * 128, activation="relu")(decoder_input)
x = layers.Reshape((16, 16, 128))(x) # Devolvemos la forma de "imagen"
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)   # Sube a 32x32
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)    # Sube a 64x64
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)    # Sube a 128x128
# Salida final con 3 canales (RGB) y sigmoid para que los píxeles estén entre 0 y 1
decoder_output = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(x)

decoder = models.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

# 6. Clase VAE Principal
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs, training=False):
        # Necesitamos implementar 'call' para poder guardar el modelo entero
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def train_step(self, data):
        # Ignoramos etiquetas si vienen en el dataset
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            recon = self.decoder(z, training=True)

            # Calculamos la diferencia al cuadrado manualmente para evitar problemas de versiones de Keras
            # Sumamos el error en los ejes ancho (1), alto (2) y canales de color (3)
            mse = tf.square(data - recon)
            recon_loss = tf.reduce_mean(tf.reduce_sum(mse, axis=[1, 2, 3]))

            # La divergencia KL se queda exactamente igual
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            )
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(), # Renombrado a 'loss' para compatibilidad con history
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# 7. Compilar y Entrenar
vae_model = VAE(encoder, decoder)
vae_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

print("Iniciando entrenamiento del VAE...")
history = vae_model.fit(dataset, epochs=EPOCHS)

# Hacemos una predicción rápida con un tensor vacío para "construir" el modelo antes de guardarlo
vae_model(tf.zeros((1, 128, 128, 3)))
# 8. Guardar y Graficar (Con el estilo de tu clase)
#vae_model.save("vae_transistor.keras")
vae_model.save_weights("vae_transistor.weights.h5")
print("Modelo guardado como 'vae_transistor.keras'.")

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history["loss"], label="Train")
plt.title("Total Loss (VAE)")
plt.xlabel("Época"); plt.ylabel("Loss")
plt.grid(True, alpha=0.3); plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history["recon_loss"], label="Train")
plt.title("Reconstruction Loss")
plt.xlabel("Época"); plt.ylabel("Loss")
plt.grid(True, alpha=0.3); plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history["kl_loss"], label="Train")
plt.title("KL Divergence Loss")
plt.xlabel("Época"); plt.ylabel("Loss")
plt.grid(True, alpha=0.3); plt.legend()

plt.tight_layout()
plt.savefig("curvas_aprendizaje_vae.png", dpi=300)
print("¡Listo! Gráfico de entrenamiento VAE guardado.")
plt.show() # Comentado para no trabar la terminal