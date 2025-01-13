import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(base_dir):
    """
    Carga las imágenes y etiquetas desde el dataset y realiza el preprocesamiento.
    Args:
        base_dir (str): Ruta al directorio del dataset.
    Returns:
        tuple: Conjuntos de entrenamiento y prueba (X_train, X_test, y_train, y_test).
    """
    images = []
    labels = []
    classes = ['car', 'motorcycle', 'truck']  # Define las clases
    for idx, category in enumerate(classes):
        category_path = os.path.join(base_dir, category)
        for file_name in os.listdir(category_path):
            img_path = os.path.join(category_path, file_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))  # Redimensiona
                images.append(img)
                labels.append(idx)
    images = np.array(images) / 255.0  # Normaliza
    labels = np.array(labels)
    return train_test_split(images, labels, test_size=0.2, random_state=42)
