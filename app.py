from flask import Flask, request, render_template
import LinealRegression

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", nombre="Francisco", edad=25, items=["Manzana", "Banana", "Naranja"])  

@app.route("/page_one")
def page_one():
    return render_template("pagina_one.html", nombre="Supervisado")

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

if __name__ == "__main__":
    app.run(debug=True)