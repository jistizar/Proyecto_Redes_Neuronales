import gradio as gr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

# --- 1. RECONSTRUIR LA ARQUITECTURA DEL VAE ---
LATENT_DIM = 128

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

# Modelo VAE
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

# --- 2. INICIALIZAR Y CARGAR PESOS ---
print("Levantando el motor VAE y cargando pesos...")
vae_model = VAE(encoder, decoder)
vae_model(tf.zeros((1, 128, 128, 3))) # Construir esqueleto
vae_model.load_weights("vae_transistor.weights.h5") # Cargar memoria

# --- 3. FUNCIÓN PRINCIPAL DE INFERENCIA ---
def procesar_imagen(img):
    # Gradio nos entrega la imagen como un array numpy. La redimensionamos a 128x128.
    img_resized = tf.image.resize(img, (128, 128)).numpy()
    img_norm = img_resized / 255.0
    input_batch = np.expand_dims(img_norm, axis=0)
    
    # Predicción (Reconstrucción)
    reconstructed_batch = vae_model.predict(input_batch, verbose=0)
    reconstructed_img = reconstructed_batch[0]
    
    # Calcular SSIM y Mapa de Error
    orig_gray = np.mean(img_norm, axis=-1)
    recon_gray = np.mean(reconstructed_img, axis=-1)
    score, mapa_similitud = ssim(orig_gray, recon_gray, data_range=1.0, full=True)
    
    mapa_error_bruto = 1.0 - mapa_similitud
    mapa_error_suavizado = gaussian_filter(mapa_error_bruto, sigma=4)
    mapa_error_norm = (mapa_error_suavizado - mapa_error_suavizado.min()) / (mapa_error_suavizado.max() - mapa_error_suavizado.min())
    
    # Binarización (Usamos tu umbral calibrado en lugar de Otsu)
    umbral_estricto = 0.65
    mascara_binaria = mapa_error_norm > umbral_estricto
    
    # Formatear imágenes de 0-1 a 0-255 para que Gradio las muestre en color
    recon_visual = np.uint8(reconstructed_img * 255)
    # Aplicar mapa de color 'jet' al calor
    heatmap_visual = np.uint8(cm.jet(mapa_error_norm)[..., :3] * 255) 
    mask_visual = np.uint8(mascara_binaria * 255)
    
    return recon_visual, heatmap_visual, mask_visual

# --- 4. INTERFAZ GRÁFICA AVANZADA CON GRADIO BLOCKS ---
print("Iniciando servidor web con Gradio Blocks...")

with gr.Blocks(theme=gr.themes.Soft(), title="Detección de Anomalías") as demo:
    
    # Encabezado centrado
    gr.Markdown("<h1 style='text-align: center;'> Sistema de Detección de Anomalías (VAE)</h1>")
    
    # Fila 1: Panel de Control (Entrada)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("###  Entrada de Datos")
            gr.Markdown("Sube la imagen del transistor capturada en la línea de ensamblaje. El modelo VAE procesará la estructura y resaltará las desviaciones.")
            input_image = gr.Image(label="Subir Transistor (Original)")
            btn_analizar = gr.Button("🔍 Ejecutar Análisis", variant="primary")
            
        with gr.Column(scale=1):
            # Dejamos esta columna vacía o para futuras métricas, 
            # así la imagen de entrada no se ve gigante y desproporcionada.
            pass 

    gr.Markdown("---")
    gr.Markdown("###  Resultados del Diagnóstico")
    
    # Fila 2: Resultados (Con todo el ancho de la pantalla para que no se corten las etiquetas)
    with gr.Row():
        out_recon = gr.Image(label="1. Reconstrucción (Sana)")
        out_heatmap = gr.Image(label="2. Mapa de Calor (SSIM)")
        out_mask = gr.Image(label="3. Máscara de Anomalía (Binarizada)")
                
    # Evento del botón
    btn_analizar.click(
        fn=procesar_imagen,
        inputs=input_image,
        outputs=[out_recon, out_heatmap, out_mask]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)