import os
from tensorflow.keras import layers, models

def build_model(input_shape=(128, 128, 3), num_classes=7):
    """
    Construye y retorna un modelo simple CNN.
    
    Args:
        input_shape (tuple): Forma de entrada de las imágenes.
        num_classes (int): Número de clases a predecir.
    
    Returns:
        model: Modelo de TensorFlow.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(train_data, val_data, model_path):
    """
    Entrena y guarda el modelo.
    
    Args:
        train_data: Datos de entrenamiento.
        val_data: Datos de validación.
        model_path (str): Ruta para guardar el modelo entrenado.
    """
    model = build_model(input_shape=(128, 128, 3), num_classes=len(train_data.class_indices))
    model.fit(train_data, validation_data=val_data, epochs=10)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
