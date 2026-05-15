from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    try:
        if request.method == "POST":

            location = request.form["location"]
            sqft = float(request.form["sqft"])
            rooms = float(request.form["rooms"])

            # ML input
            features = np.array([[sqft, rooms]])

            price = model.predict(features)[0]

            prediction = round(price, 2)

    except:
        error = "Invalid input or server error"

    return render_template("index.html",
                           prediction=prediction,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)