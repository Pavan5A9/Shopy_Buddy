from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load ML model
model = joblib.load("fraud_model.pkl")

# Load dataset safely
dataset_path = "ecommerce_return_refund_fraud_dataset_100k.csv"

if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
else:
    df = None


# ---------------- HOME PAGE ----------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------- PREDICTION PAGE ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = None

    if request.method == "POST":

        order_value = float(request.form["order_value"])
        return_count = int(request.form["return_count"])
        customer_age = int(request.form["customer_age"])
        days_to_return = int(request.form["days_to_return"])

        features = np.array([[order_value, return_count, customer_age, days_to_return]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "High Risk Fraud"
        else:
            result = "Low Risk"

    return render_template("predict.html", result=result)


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():

    if df is None:
        return "Dataset not found"

    total_orders = len(df)
    fraud_cases = df["fraud_flag"].sum()
    fraud_rate = round((fraud_cases / total_orders) * 100, 2)

    return render_template(
        "dashboard.html",
        total_orders=total_orders,
        fraud_cases=fraud_cases,
        fraud_rate=fraud_rate
    )


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
