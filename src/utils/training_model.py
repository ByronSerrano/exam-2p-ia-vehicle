import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2

def train_model(X_train, y_train, X_test, y_test):
    """
    Define y entrena un modelo basado en MobileNetV2.
    Args:
        X_train, y_train: Datos de entrenamiento.
        X_test, y_test: Datos de prueba.
    Returns:
        tuple: Modelo entrenado y su historial de entrenamiento.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # Tres clases: car, motorcycle, truck
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
    return model, history
