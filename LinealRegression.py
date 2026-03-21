import pandas as pd
import numpy as np 
import io
import base64
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento
data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df = pd.DataFrame(data)
X = df[["Study Hours"]]
y = df["Final Grade"]

# Entrenar el modelo
model = LinearRegression()
model.fit(X, y)

def predict_grade(study_hours):
    """
    Predice la nota basada en horas de estudio
    """
    result = model.predict([[study_hours]])[0]
    # Limitar la nota entre 0 y 5 (asumiendo escala 0-5)
    result = max(0, min(5, result))
    return round(result, 2)  # Redondear a 2 decimales

def generate_regression_plot(study_hours, predicted_grade):
    """
    Genera una gráfica de regresión lineal mostrando:
    - Puntos de datos reales
    - Línea de regresión
    - Punto de predicción
    """
    # Crear la figura
    plt.figure(figsize=(12, 8))
    
    # 1. Graficar puntos de datos reales
    plt.scatter(df['Study Hours'], df['Final Grade'], 
                color='blue', alpha=0.6, s=100, 
                label='Datos Reales', edgecolors='darkblue', linewidth=2)
    
    # 2. Generar línea de regresión
    x_range = np.linspace(0, 22, 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    
    plt.plot(x_range, y_range, 'red', linewidth=3, 
             label=f'Línea de Regresión (y = {model.coef_[0]:.2f}x + {model.intercept_:.2f})')
    
    # 3. Marcar el punto predicho
    plt.scatter([study_hours], [predicted_grade], 
                color='green', s=200, marker='*', 
                label=f'Predicción: {predicted_grade} puntos', 
                edgecolors='darkgreen', linewidth=2, zorder=5)
    
    # 4. Líneas auxiliares para el punto predicho
    plt.axvline(x=study_hours, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=predicted_grade, color='gray', linestyle='--', alpha=0.5)
    
    # Personalizar la gráfica
    plt.xlabel('Horas de Estudio', fontsize=14, fontweight='bold')
    plt.ylabel('Nota Final (0-5 escala)', fontsize=14, fontweight='bold')
    plt.title('Regresión Lineal: Relación Horas de Estudio vs Nota Final', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='upper left')
    
    # Limitar ejes para mejor visualización
    plt.xlim(0, 22)
    plt.ylim(0, 6)
    
    # Añadir texto con estadísticas
    stats_text = f"R² = {model.score(X, y):.3f}\n"
    stats_text += f"Correlación: {np.corrcoef(df['Study Hours'], df['Final Grade'])[0,1]:.3f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Guardar la gráfica en un objeto bytes
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    
    # Convertir a base64 para incrustar en HTML
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

# Función para obtener estadísticas del modelo
def get_model_stats():
    """
    Retorna estadísticas del modelo
    """
    return {
        'coefficient': round(model.coef_[0], 3),
        'intercept': round(model.intercept_, 3),
        'r2_score': round(model.score(X, y), 3),
        'correlation': round(np.corrcoef(df['Study Hours'], df['Final Grade'])[0,1], 3)
    }