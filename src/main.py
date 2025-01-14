from utils.load_dataset import load_dataset
from utils.training_model import train_and_save_model
from utils.process_video import process_video

if __name__ == "__main__":
    # Rutas
    train_path = "src/data/cars_dataset/train"
    test_path = "src/data/cars_dataset/test"
    model_path = "src/model/model_ia.h5"
    model_detection_object = "src/model/multi_detection_object/yolov8n.pt"
    video_path = "src/results/input/video_input.mp4"
    output_path = "src/results/output/video_output.mp4"

    # Verificar rutas
    import os
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"La ruta {train_path} no existe. Verifica la estructura de tu proyecto.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"La ruta {test_path} no existe. Verifica la estructura de tu proyecto.")
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Cargar datasets
    print("Cargando datasets...")
    train_data = load_dataset(train_path)
    test_data = load_dataset(test_path)

    # Entrenar y guardar el modelo
    print("Entrenando el modelo...")
    train_and_save_model(train_data, test_data, model_path)

    # Procesar el video
    print("Procesando el video...")
    process_video(video_path, model_detection_object, model_path, output_path)

    print("Proceso completado. Video guardado en:", output_path)
