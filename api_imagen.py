from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import io
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

# --- 1. RECONSTRUIR LA ARQUITECTURA DEL VAE ---
LATENT_DIM = 128

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

# Encoder y Decoder
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

decoder_input = layers.Input(shape=(LATENT_DIM,))
x = layers.Dense(16 * 16 * 128, activation="relu")(decoder_input)
x = layers.Reshape((16, 16, 128))(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
decoder_output = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(x)
decoder = models.Model(decoder_input, decoder_output)

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

# --- 2. INICIALIZAR VAE Y CARGAR PESOS ---
print("Levantando motor VAE...")
vae_model = VAE(encoder, decoder)
vae_model(tf.zeros((1, 128, 128, 3))) 
vae_model.load_weights("vae_transistor.weights.h5")

# --- 3. CREAR LA API REST CON FASTAPI ---
app = FastAPI(title="API Detección de Anomalías (Visión Artificial)", version="1.0")

@app.get("/")
def estado_servidor():
    return {"mensaje": "Servidor VAE en línea y esperando imágenes industriales."}

# Usamos POST porque estamos "Enviando" datos pesados (la imagen) al servidor
@app.post("/diagnosticar_imagen")
async def diagnosticar(file: UploadFile = File(...)):
    # 1. Leer la imagen enviada por el cliente
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. Preprocesar (Igual que en nuestro script de evaluación)
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    input_batch = np.expand_dims(img_array, axis=0)
    
    # 3. Predicción del modelo
    reconstructed = vae_model.predict(input_batch, verbose=0)[0]
    
    # 4. Calcular el error estructural (SSIM)
    orig_gray = np.mean(img_array, axis=-1)
    recon_gray = np.mean(reconstructed, axis=-1)
    score, mapa_similitud = ssim(orig_gray, recon_gray, data_range=1.0, full=True)
    
    mapa_error = 1.0 - mapa_similitud
    mapa_error_suavizado = gaussian_filter(mapa_error, sigma=4)
    # Evitar división por cero si la imagen es perfectamente idéntica
    rango = mapa_error_suavizado.max() - mapa_error_suavizado.min()
    if rango == 0:
        mapa_error_norm = mapa_error_suavizado
    else:
        mapa_error_norm = (mapa_error_suavizado - mapa_error_suavizado.min()) / rango
    
    # 5. Binarización estricta
    umbral_estricto = 0.65
    mascara = mapa_error_norm > umbral_estricto
    
    # 6. Toma de decisión (Lógica de negocio)
    porcentaje_defecto = float(np.mean(mascara) * 100)
    # Si más del 0.5% de la pieza tiene daños severos, la descartamos
    es_defectuoso = porcentaje_defecto > 0.5 
    
    # 7. Retornar el diagnóstico en formato JSON estándar
    return {
        "archivo_analizado": file.filename,
        "estado": "Defectuoso ❌" if es_defectuoso else "Sano ✅",
        "porcentaje_area_anomala": round(porcentaje_defecto, 2),
        "indice_similitud_general": round(float(score), 4)
    }