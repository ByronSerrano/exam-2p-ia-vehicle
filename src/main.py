from utils.load_dataset import load_dataset
from utils.training_model import train_model
from utils.process_video import process_video
from utils.pdf.generate_pdf import create_pdf

path_to_pdf = "../results/pdf/report.pdf"
path_to_video = "../video_test/"

def main():
    # Cargar dataset
    print("Cargando el dataset...")
    X_train, X_test, y_train, y_test = load_dataset("./data/OIDv6/train")
    
    # Entrenar el modelo
    print("Entrenando el modelo...")
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    # Procesar el video
    print("Procesando el video...")
    process_video("input_video.mp4", model)
    
    # Generar reporte
    print("Generando el reporte...")
    create_pdf(history, path_to_pdf)
    
    print("¡Proceso completado!")

if __name__ == "__main__":
    main()
