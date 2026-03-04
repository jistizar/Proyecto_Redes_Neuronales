import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu

# 1. Configuración
IMG_SIZE = (128, 128)
TEST_IMG_PATH = "./dataset/transistor/test/bent_lead/000.png" 

print("Cargando el modelo...")
autoencoder = tf.keras.models.load_model("autoencoder_transistor.keras")

def load_and_preprocess_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = tf.keras.utils.img_to_array(img) / 255.0  
    img_batch = np.expand_dims(img_array, axis=0) 
    return img_array, img_batch

original_img, input_batch = load_and_preprocess_image(TEST_IMG_PATH)

# 2. Reconstruir
reconstructed_batch = autoencoder.predict(input_batch)
reconstructed_img = reconstructed_batch[0]

# 3. SSIM y Mapa de Error Suavizado
orig_gray = np.mean(original_img, axis=-1)
recon_gray = np.mean(reconstructed_img, axis=-1)
score, mapa_similitud = ssim(orig_gray, recon_gray, data_range=1.0, full=True)

mapa_error_bruto = 1.0 - mapa_similitud
mapa_error_suavizado = gaussian_filter(mapa_error_bruto, sigma=4)
mapa_error_norm = (mapa_error_suavizado - mapa_error_suavizado.min()) / (mapa_error_suavizado.max() - mapa_error_suavizado.min())

# 4. EL TOQUE FINAL: Binarización con Otsu
# Calculamos el umbral perfecto matemáticamente
umbral = threshold_otsu(mapa_error_norm)
# Creamos la máscara (True/False que se traduce a Blanco/Negro)
mascara_binaria = mapa_error_norm > umbral

# 5. Visualización de 4 paneles para la presentación
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Original (Defecto)")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Reconstrucción")
plt.imshow(reconstructed_img)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Mapa de Calor")
plt.imshow(mapa_error_norm, cmap='jet') 
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Máscara Binaria (Otsu)")
plt.imshow(mascara_binaria, cmap='gray') 
plt.axis('off')

plt.tight_layout()
plt.savefig("resultado_anomalia_otsu.png", dpi=300)
print("¡Listo! Imagen 'resultado_anomalia_otsu.png' guardada con éxito.")

# Comentamos esta línea para que tu terminal no se quede bloqueada
plt.show()