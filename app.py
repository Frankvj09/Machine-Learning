from flask import Flask

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", nombre="Francisco", edad=25, items=["Manzana", "Banana", "Naranja"])  

@app.route("/page_one")
def page_one():
    return render_template("pagina_one.html", nombre="Supervisado")

if __name__ == "__main__":
    app.run(debug=True)
