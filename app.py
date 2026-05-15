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

            features = np.array([[sqft, rooms]])
            price = model.predict(features)[0]

            # PROFESSIONAL FORMATTING
            prediction = f"${price:,.2f}"

    except ValueError:
        error = "Please enter valid numeric values."
    except Exception:
        error = "Server error. Please try again."

    return render_template("index.html",
                           prediction=prediction,
                           error=error)


if __name__ == "__main__":
    app.run(debug=True)