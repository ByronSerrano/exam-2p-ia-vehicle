import cv2
import numpy as np
from ultralytics import YOLO

def process_video(video_path, yolo_model_path, classifier_model_path, output_path):
    """
    Procesa un video usando YOLO para detección y un modelo personalizado para clasificación de marcas.
    
    Args:
        video_path (str): Ruta al video de entrada.
        yolo_model_path (str): Ruta al modelo YOLO.
        classifier_model_path (str): Ruta al modelo entrenado para clasificación.
        output_path (str): Ruta para guardar el video procesado.
    """
    # Cargar el modelo YOLO
    yolo_model = YOLO(yolo_model_path)

    # Cargar el modelo de clasificación
    from tensorflow.keras.models import load_model
    classifier_model = load_model(classifier_model_path)
    labels = {0: 'Audi', 1: 'Hyundai Creta', 2: 'Mahindra', 3: 'Rolls Royce',
              4: 'Swift', 5: 'Tata Safari', 6: 'Toyota'}

    # Abrir el video de entrada
    cap = cv2.VideoCapture(video_path)
    
    # Configurar la salida del video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detección de vehículos con YOLO
        results = yolo_model(frame)
        
        for obj in results[0].boxes:
            # Obtener coordenadas de detección
            coords = obj.xyxy[0].cpu().numpy()  # Convierte las coordenadas a un array NumPy
            x1, y1, x2, y2 = map(int, coords)

            # Recortar la región detectada
            detected_region = frame[y1:y2, x1:x2]
            if detected_region.size == 0:
                continue

            # Redimensionar la región para el modelo de clasificación
            resized_region = cv2.resize(detected_region, (128, 128)) / 255.0
            resized_region = np.expand_dims(resized_region, axis=0)

            # Clasificar la marca del vehículo
            prediction = classifier_model.predict(resized_region)
            label = labels[np.argmax(prediction)]

            # Dibujar el rectángulo y la etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Guardar el cuadro procesado
        out.write(frame)
        cv2.imshow("Detecciones", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
