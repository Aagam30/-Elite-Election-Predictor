from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("election_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = ""
    if request.method == "POST":
        try:
            age = float(request.form["age"])
            income = float(request.form["income"])
            education = float(request.form["education"])
            gender = 1 if request.form["gender"] == "Male" else 0
            region = int(request.form["region"])

            features = np.array([[age, income, education, gender, region]])
            scaled = scaler.transform(features)
            result = model.predict(scaled)

            prediction = "Yes" if result[0] == 1 else "No"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)