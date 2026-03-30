import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Dataset desde Excel
df = pd.read_excel('dataset_knn_clientes_2000.xlsx')

X = df[['edad', 'ingresos']]
y = df['compra']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_scaled, y)

def predecir_knn(edad, ingresos):
    nuevo = np.array([[edad, ingresos]])
    nuevo_scaled = scaler.transform(nuevo)

    pred = modelo.predict(nuevo_scaled)[0]
    prob = modelo.predict_proba(nuevo_scaled)[0]

    # GRÁFICA
    plt.figure()

    # Datos originales
    plt.scatter(X['edad'], X['ingresos'], c=y)

    # Punto nuevo
    plt.scatter(edad, ingresos, marker='x', s=200)

    plt.xlabel("Edad")
    plt.ylabel("Ingresos")
    plt.title("KNN - Clasificación")

    # Guardar imagen
    ruta = "static/knn_plot.png"
    plt.savefig(ruta)
    plt.close()

    return pred, prob, ruta