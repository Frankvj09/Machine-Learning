import pandas as pd
import numpy as np 
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import os

# Obtener la ruta del archivo CSV
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'dataset_regresion_logistica.csv')

# Cargar los datos
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    # Si no encuentra el archivo, crear datos de ejemplo
    print("Archivo no encontrado. Creando datos de ejemplo...")
    np.random.seed(42)
    n_samples = 200
    data = pd.DataFrame({
        'edad': np.random.randint(18, 65, n_samples),
        'ingreso_mensual': np.random.randint(1000000, 7000000, n_samples),
        'visitas_web_mes': np.random.randint(1, 21, n_samples),
        'tiempo_sitio_min': np.random.uniform(1, 25, n_samples),
        'compras_previas': np.random.randint(0, 16, n_samples),
        'descuento_usado': np.random.randint(0, 2, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })

# Explorar el conjunto de datos
X = data.drop('target', axis=1)  # Variables independientes
y = data['target']  # Variable dependiente (0 = no compra, 1 = compra)

# Dividir el dataset en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos (recomendable para regresión logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de Regresión Logística
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calcular exactitud
accuracy = accuracy_score(y_test, y_pred)

def predict_purchase(features):
    """
    Predice si un cliente realizará una compra basado en sus características
    
    Args:
        features: dict con las siguientes claves:
            - edad: int
            - ingreso_mensual: float
            - visitas_web_mes: int
            - tiempo_sitio_min: float
            - compras_previas: int
            - descuento_usado: int (0 o 1)
    
    Returns:
        dict con predicción, probabilidad y mensaje
    """
    try:
        # Crear array con las características en el orden correcto
        feature_array = np.array([[
            features['edad'],
            features['ingreso_mensual'],
            features['visitas_web_mes'],
            features['tiempo_sitio_min'],
            features['compras_previas'],
            features['descuento_usado']
        ]])
        
        # Estandarizar los datos de entrada
        feature_scaled = scaler.transform(feature_array)
        
        # Predecir clase y probabilidad
        prediction = model.predict(feature_scaled)[0]
        probability = model.predict_proba(feature_scaled)[0][1]
        
        return {
            'purchase': bool(prediction),
            'probability': round(probability, 3),
            'message': '✅ WILL PURCHASE' if prediction == 1 else '❌ WILL NOT PURCHASE',
            'confidence': probability if prediction == 1 else 1 - probability
        }
    except Exception as e:
        return {
            'purchase': False,
            'probability': 0,
            'message': f'Error: {str(e)}',
            'confidence': 0
        }

def generate_logistic_plot(features, prediction):
    """
    Genera gráficas de regresión logística con visualizaciones
    """
    # Crear figura con 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Gráfico de distribución de compras por edad
    ax1 = axes[0, 0]
    purchase_ages = data[data['target'] == 1]['edad']
    no_purchase_ages = data[data['target'] == 0]['edad']
    
    ax1.hist(purchase_ages, bins=15, alpha=0.7, color='green', label='Will Purchase', edgecolor='darkgreen')
    ax1.hist(no_purchase_ages, bins=15, alpha=0.7, color='red', label='Will Not Purchase', edgecolor='darkred')
    ax1.axvline(x=features['edad'], color='blue', linestyle='--', linewidth=2, label=f'Customer Age: {features["edad"]}')
    ax1.set_xlabel('Age', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Age Distribution by Purchase Decision', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gráfico de ingresos vs tiempo en sitio
    ax2 = axes[0, 1]
    purchase_data = data[data['target'] == 1]
    no_purchase_data = data[data['target'] == 0]
    
    ax2.scatter(purchase_data['ingreso_mensual'], purchase_data['tiempo_sitio_min'], 
                color='green', alpha=0.6, label='Will Purchase', s=50)
    ax2.scatter(no_purchase_data['ingreso_mensual'], no_purchase_data['tiempo_sitio_min'], 
                color='red', alpha=0.6, label='Will Not Purchase', s=50)
    ax2.scatter([features['ingreso_mensual']], [features['tiempo_sitio_min']], 
                color='blue', s=200, marker='*', label='Customer', edgecolors='darkblue', linewidth=2)
    ax2.set_xlabel('Monthly Income', fontsize=12)
    ax2.set_ylabel('Time on Site (min)', fontsize=12)
    ax2.set_title('Income vs Time on Site', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Matriz de confusión
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3)
    ax3.set_xlabel('Predicted', fontsize=12)
    ax3.set_ylabel('Actual', fontsize=12)
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(['No Purchase', 'Purchase'])
    ax3.set_yticklabels(['No Purchase', 'Purchase'])
    
    # 4. Curva ROC
    ax4 = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax4.scatter([1 - prediction['probability'] if not prediction['purchase'] else prediction['probability']], 
                [prediction['probability'] if prediction['purchase'] else 1 - prediction['probability']], 
                color='blue', s=200, marker='*', label=f'Customer: {prediction["message"]}', zorder=5)
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontsize=12)
    ax4.set_title(f'ROC Curve (AUC = {roc_auc:.3f})', fontsize=14, fontweight='bold')
    ax4.legend(loc="lower right")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar la gráfica
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def get_model_stats():
    """
    Retorna estadísticas del modelo de regresión logística
    """
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'accuracy': round(accuracy, 3),
        'precision': round(report['1']['precision'], 3) if '1' in report else 0,
        'recall': round(report['1']['recall'], 3) if '1' in report else 0,
        'f1_score': round(report['1']['f1-score'], 3) if '1' in report else 0,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(data),
        'purchase_samples': int(data['target'].sum()),
        'no_purchase_samples': int(len(data) - data['target'].sum()),
        'features': list(X.columns)
    }

def get_feature_importance():
    """
    Retorna la importancia de las características (coeficientes)
    """
    coefficients = model.coef_[0]
    feature_names = X.columns
    
    # Ordenar por importancia
    importance = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
    
    return importance

# Función para obtener el scaler (si se necesita en otros lugares)
def get_scaler():
    return scaler

# Función para obtener el modelo entrenado
def get_model():
    return model