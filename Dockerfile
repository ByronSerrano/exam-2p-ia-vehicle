FROM tensorflow/tensorflow:2.12.0-gpu
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el modelo y el código
COPY models/ ./models/
COPY src/ ./src/

# Comando por defecto
CMD ["python", "./src/main.py"]
