from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load trained Random Forest model
model = joblib.load("rf_fraud_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob = None

    if request.method == "POST":
        try:
            # ---- Collect inputs ----
            type_ = request.form["type"]
            amount = float(request.form["amount"])
            old_org = float(request.form["old_org"])
            new_org = float(request.form["new_org"])
            old_dest = float(request.form["old_dest"])
            new_dest = float(request.form["new_dest"])

            # ---- Feature engineering ----
            error_org = old_org - new_org - amount
            error_dest = new_dest - old_dest - amount
            balance_diff_org = new_org - old_org
            balance_diff_dest = new_dest - old_dest
            ratio_org = new_org / (old_org + 1)
            ratio_dest = new_dest / (old_dest + 1)
            log_amount = np.log1p(amount)

            # ---- One-hot encoding ----
            type_features = {"CASH_OUT": 0, "DEBIT": 0, "PAYMENT": 0, "TRANSFER": 0}
            if type_ in type_features:
                type_features[type_] = 1

            # ---- Default step ----
            step = 0

            # ---- Feature order (19 total) ----
            features = [
                step, amount, old_org, new_org, old_dest, new_dest,
                error_org, error_dest, balance_diff_org, balance_diff_dest,
                ratio_org, ratio_dest,
                type_features["CASH_OUT"],
                type_features["DEBIT"],
                type_features["PAYMENT"],
                type_features["TRANSFER"],
                log_amount, 0, 0
            ]

            X_input = np.array([features])

            # ---- Base ML Prediction ----
            base_prob = model.predict_proba(X_input)[0][1]
            risk = base_prob

            # ---- ✅ Hybrid Rule-Based Enhancements ----
            if type_ in ["CASH_OUT", "TRANSFER", "DEBIT", "PAYMENT"]:
                if amount > 10000000:
                    risk += 0.25
                if new_org <= 0 and old_org > 0:
                    risk += 0.35
                if abs(new_dest - old_dest) < 1e-2:
                    risk += 0.40
                if amount > old_org:
                    risk += 0.45
                if abs((old_org - new_org) - amount) > 1e-2:
                    risk += 0.30
                if type_ == "TRANSFER" and abs((new_dest - old_dest) - amount) > 1e-2:
                    risk += 0.30
                if new_dest == 0 and old_dest == 0:
                    risk += 0.25

            # Cap risk at 1.0
            risk = min(risk, 1.0)
            prob = round(risk * 100, 2)

            # ---- Final classification (3-tier) ----
            if risk > 0.7:
                prediction = "🚨 High Risk Fraud"
            elif risk > 0.3:
                prediction = "⚠️ Suspicious Transaction"
            else:
                prediction = "✅ Legitimate Transaction"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", result=prediction, prob=prob)

if __name__ == "__main__":
    app.run(debug=True, port=8000)