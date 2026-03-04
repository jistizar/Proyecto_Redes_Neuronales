import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# 1. Configuración
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30
DATA_DIR = "./dataset/transistor/train"

print(f"Cargando imágenes de: {DATA_DIR}...")

# 2. Cargar y preprocesar los datos
dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels=None, 
    color_mode="rgb",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
dataset = dataset.map(lambda x: (normalization_layer(x), normalization_layer(x))) 

# 3. Construir el Autoencoder Convolucional
def build_autoencoder():
    input_img = layers.Input(shape=(128, 128, 3))
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

ae_model = build_autoencoder()
ae_model.summary()

# 4. Entrenar el modelo
print("Iniciando entrenamiento del Autoencoder...")
history = ae_model.fit(dataset, epochs=EPOCHS)

# 5. Guardar el modelo
ae_model.save("autoencoder_transistor.keras")
print("Modelo guardado como 'autoencoder_transistor.keras'.")

# 6. Generar la curva de aprendizaje
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Pérdida (MSE)', color='blue', linewidth=2)
plt.title('Curva de Aprendizaje - Autoencoder (Transistor)')
plt.ylabel('Error Cuadrático Medio (MSE)')
plt.xlabel('Época')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("curva_aprendizaje.png", dpi=300)
print("¡Listo! Gráfico guardado como 'curva_aprendizaje.png'.")