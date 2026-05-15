from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")


# convert form string → numeric score
def form_to_score(form):

    mapping = {
        "W": 1,
        "D": 0.5,
        "L": 0
    }

    return sum(mapping[f] for f in form.upper()) / 5


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        home_form = request.form["home_form"]
        away_form = request.form["away_form"]

        home_score = form_to_score(home_form)
        away_score = form_to_score(away_form)

        diff = home_score - away_score

        pred = model.predict([[diff]])[0]
        prob = model.predict_proba([[diff]])[0][1]

        prediction = "HOME WIN" if pred == 1 else "AWAY/DRAW RISK"
        probability = round(prob * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability)

if __name__ == "__main__":
    app.run(debug=True)