from fpdf import FPDF

def create_pdf(history, output_path):
    """
    Genera un reporte en formato PDF con las métricas del modelo.
    Args:
        history: Historial del entrenamiento del modelo.
        output_path (str): Ruta para guardar el PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Reporte de Entrenamiento", ln=True, align="C")

    # Agregar datos del historial (precisión y pérdida)
    pdf.cell(200, 10, txt="Métricas del Modelo:", ln=True, align="L")
    for key, values in history.history.items():
        pdf.cell(200, 10, txt=f"{key}: {values[-1]:.4f}", ln=True, align="L")

    pdf.output(output_path)
    print(f"Reporte guardado en {output_path}")
