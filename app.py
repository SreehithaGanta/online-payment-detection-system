from flask import Flask, render_template, request
import numpy as np
import pickle

# ✅ CREATE FLASK APP FIRST
app = Flask(__name__)

# ✅ LOAD MODEL AFTER APP IS CREATED
model = pickle.load(open("payments.pkl", "rb"))
print("MODEL EXPECTS FEATURES:", model.n_features_in_)


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("home.html")

# ---------------- PREDICT PAGE + LOGIC ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = [
            float(request.form["step"]),
            float(request.form["type"]),
            float(request.form["amount"]),
            float(request.form["oldbalanceOrg"]),
            float(request.form["newbalanceOrg"]),
            float(request.form["oldbalanceDest"]),
            float(request.form["newbalanceDest"]),
        ]

        final_input = np.array(data).reshape(1, -1)
        prediction = model.predict(final_input)[0]

        result = "⚠️ Fraudulent Transaction" if prediction == 1 else "✅ Not Fraudulent Transaction"

        return render_template("submit.html", prediction_text=result)

    # GET request → show form
    return render_template("predict.html")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)
