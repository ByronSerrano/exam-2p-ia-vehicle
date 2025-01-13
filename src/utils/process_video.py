import cv2
import numpy as np

def process_video(video_path, model):
    """
    Procesa un video, clasifica vehículos en cada cuadro y muestra los resultados en tiempo real.
    Args:
        video_path (str): Ruta al video de entrada.
        model: Modelo entrenado para la clasificación.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (224, 224))
        input_data = np.expand_dims(resized_frame / 255.0, axis=0)
        predictions = model.predict(input_data)
        label = np.argmax(predictions)

        # Mostrar resultados en el cuadro
        cv2.putText(frame, f"Class: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
