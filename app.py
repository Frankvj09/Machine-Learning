from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
import LinealRegression
import LogisticRegression

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html", nombre="Francisco", edad=25, items=["Manzana", "Banana", "Naranja"])  

@app.route("/page_one")
def page_one():
    return render_template("pagina_one.html", nombre="Supervisado")
#_______________________________________________________________
#Regresión Lineal
@app.route("/LinealRegression", methods=["GET", "POST"])
def predict_grade():
    if request.method == "POST":
        try:
            study_hours = request.form.get("study_hours")
            
            if study_hours is None or study_hours == "":
                return render_template("lineaRegression.html", 
                                     error="Por favor ingresa las horas de estudio")
            
            study_hours = float(study_hours)
            
            # Validar rango de horas
            if study_hours < 0 or study_hours > 24:
                return render_template("lineaRegression.html", 
                                     error="Las horas deben estar entre 0 y 24")
            
            # Obtener predicción
            result = LinealRegression.predict_grade(study_hours)
            
            # Generar gráfica
            plot_url = LinealRegression.generate_regression_plot(study_hours, result)
            
            # Obtener estadísticas del modelo
            stats = LinealRegression.get_model_stats()
            
            return render_template("lineaRegression.html", 
                                 result=result, 
                                 plot_url=plot_url,
                                 hours=study_hours,
                                 stats=stats)
            
        except ValueError:
            return render_template("lineaRegression.html", 
                                 error="Por favor ingresa un número válido")
        except Exception as e:
            return render_template("lineaRegression.html", 
                                 error=f"Error: {str(e)}")
    
    # Método GET - mostrar formulario
    stats = LinealRegression.get_model_stats()
    return render_template("lineaRegression.html", 
                         result=None,
                         stats=stats)
#_______________________________________________________________

# LogisticRegression

@app.route("/LogisticRegression", methods=["GET", "POST"])
def logistic_regression():
    if request.method == "POST":
        try:
            # Obtener datos del formulario
            features = {
                'edad': float(request.form.get('edad', 0)),
                'ingreso_mensual': float(request.form.get('ingreso_mensual', 0)),
                'visitas_web_mes': float(request.form.get('visitas_web_mes', 0)),
                'tiempo_sitio_min': float(request.form.get('tiempo_sitio_min', 0)),
                'compras_previas': float(request.form.get('compras_previas', 0)),
                'descuento_usado': float(request.form.get('descuento_usado', 0))
            }
            
            # Validaciones básicas
            if features['edad'] < 18 or features['edad'] > 100:
                return render_template("logistic_regression_app.html", 
                                     error="Age must be between 18 and 100 years")
            
            if features['ingreso_mensual'] < 0:
                return render_template("logistic_regression_app.html", 
                                     error="Income cannot be negative")
            
            # Obtener predicción
            prediction = LogisticRegression.predict_purchase(features)
            
            # Generar gráfica
            plot_url = LogisticRegression.generate_logistic_plot(features, prediction)
            
            # Obtener estadísticas del modelo
            stats = LogisticRegression.get_model_stats()
            feature_importance = LogisticRegression.get_feature_importance()
            
            return render_template("logistic_regression_app.html", 
                                 result=prediction,
                                 plot_url=plot_url,
                                 form_data=features,
                                 stats=stats,
                                 feature_importance=feature_importance)
            
        except ValueError as e:
            return render_template("logistic_regression_app.html", 
                                 error=f"Please enter valid numbers: {str(e)}")
        except Exception as e:
            return render_template("logistic_regression_app.html", 
                                 error=f"Error: {str(e)}")
    
    # Método GET - mostrar formulario
    stats = LogisticRegression.get_model_stats()
    feature_importance = LogisticRegression.get_feature_importance()
    return render_template("logistic_regression_app.html", 
                         result=None,
                         stats=stats,
                         feature_importance=feature_importance)
#____________________________________________________________
# Regresion Model KNN

@app.route('/knn-concepts')
def knn_concepts():
    return render_template('knn_concepts.html')

from KNN import predecir_knn

@app.route('/knn', methods=['GET', 'POST'])
def knn():
    resultado = None
    probabilidad = None
    grafica = None

    if request.method == 'POST':
        edad = float(request.form['edad'])
        ingresos = float(request.form['ingresos'])

        pred, prob, ruta = predecir_knn(edad, ingresos)

        resultado = "Comprará" if pred == 1 else "No comprará"
        probabilidad = prob
        grafica = ruta

    return render_template('knn.html',
                           resultado=resultado,
                           probabilidad=probabilidad,
                           grafica=grafica)
# _________________________________________________________________________

# Use Cases Routes
@app.route('/netflix')
def use_case_netflix():
    return render_template('/netflix.html')

@app.route('/amazon')
def use_case_amazon():
    return render_template('/amazon.html')

@app.route('/tesla')
def tesla():
    return render_template('/tesla.html')

@app.route('/linear-concepts')
def linear_concepts():
    return render_template('linear_concepts.html')

@app.route('/logistic-concepts')
def logistic_concepts():
    return render_template('logistic_concepts.html')

if __name__ == "__main__":
    app.run(debug=True)