# 1. Usar una imagen base ligera de Python
FROM python:3.10-slim

# 2. Establecer la carpeta de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar la lista de dependencias e instalarlas con tolerancia a fallos de red
COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries=10 --no-cache-dir -r requirements.txt

# 4. Copiar todo el código, imágenes y pesos de tu proyecto al contenedor
COPY . .

# 5. Abrir el puerto 7860 para que Gradio pueda comunicarse con el exterior
EXPOSE 7860

# 6. El comando que encenderá el sistema cuando el contenedor inicie
CMD ["python", "app.py"]