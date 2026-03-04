import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu

# 1. Configuración
IMG_SIZE = (128, 128)
LATENT_DIM = 128
TEST_IMG_PATH = "./dataset/transistor/test/bent_lead/000.png" 

# 2. Reconstruir la Arquitectura del VAE
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

# Encoder
encoder_input = layers.Input(shape=(128, 128, 3))
x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_input)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
z_mean = layers.Dense(LATENT_DIM)(x)
z_log_var = layers.Dense(LATENT_DIM)(x)
z = Sampling()([z_mean, z_log_var])
encoder = models.Model(encoder_input, [z_mean, z_log_var, z])

# Decoder
decoder_input = layers.Input(shape=(LATENT_DIM,))
x = layers.Dense(16 * 16 * 128, activation="relu")(decoder_input)
x = layers.Reshape((16, 16, 128))(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
decoder_output = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(x)
decoder = models.Model(decoder_input, decoder_output)

# Clase VAE
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

vae_model = VAE(encoder, decoder)
# Construimos el esqueleto y cargamos la "memoria" (los pesos)
vae_model(tf.zeros((1, 128, 128, 3)))
vae_model.load_weights("vae_transistor.weights.h5")
print("Modelo VAE cargado exitosamente.")

# 3. Cargar imagen de prueba
def load_and_preprocess_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = tf.keras.utils.img_to_array(img) / 255.0  
    img_batch = np.expand_dims(img_array, axis=0) 
    return img_array, img_batch

original_img, input_batch = load_and_preprocess_image(TEST_IMG_PATH)

# 4. Reconstrucción con el VAE
reconstructed_batch = vae_model.predict(input_batch)
reconstructed_img = reconstructed_batch[0]

# 5. SSIM y Mapa de Error Suavizado
orig_gray = np.mean(original_img, axis=-1)
recon_gray = np.mean(reconstructed_img, axis=-1)
score, mapa_similitud = ssim(orig_gray, recon_gray, data_range=1.0, full=True)

mapa_error_bruto = 1.0 - mapa_similitud
mapa_error_suavizado = gaussian_filter(mapa_error_bruto, sigma=4)
mapa_error_norm = (mapa_error_suavizado - mapa_error_suavizado.min()) / (mapa_error_suavizado.max() - mapa_error_suavizado.min())

# 6. Binarización Manual / Percentil (Reemplazando a Otsu)
# Como el mapa_error_norm va de 0 a 1, y el defecto es muy rojo (alto), 
# cortamos todo lo que esté por debajo de un nivel estricto, por ejemplo, 0.6 (60%)
umbral_estricto = 0.65 
mascara_binaria = mapa_error_norm > umbral_estricto

# 7. Visualización
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Original (Defecto)")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Reconstrucción (VAE)")
plt.imshow(reconstructed_img)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Mapa de Calor")
plt.imshow(mapa_error_norm, cmap='jet') 
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Máscara (Otsu)")
plt.imshow(mascara_binaria, cmap='gray') 
plt.axis('off')

plt.tight_layout()
plt.savefig("resultado_vae_otsu.png", dpi=300)
print("¡Listo! Imagen 'resultado_vae_otsu2.png' guardada con éxito.")