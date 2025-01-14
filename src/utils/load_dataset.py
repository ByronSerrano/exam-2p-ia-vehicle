import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(dataset_path, img_size=(128, 128), batch_size=32):
    """
    Carga el dataset desde una carpeta estructurada.
    
    Args:
        dataset_path (str): Ruta al dataset (train o test).
        img_size (tuple): Tamaño al que redimensionar las imágenes.
        batch_size (int): Tamaño de los lotes para el generador.
    
    Returns:
        Data generator: Generador de datos para el modelo.
    """
    datagen = ImageDataGenerator(rescale=1.0/255)
    dataset = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    return dataset
