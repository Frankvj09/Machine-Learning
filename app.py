from flask import Flask

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", nombre="Francisco", edad=25, items=["Manzana", "Banana", "Naranja"])



if __name__ == "__main__":
    app.run(debug=True)
