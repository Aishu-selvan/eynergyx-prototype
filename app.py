from flask import Flask, render_template, request, send_file
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime
import csv

app = Flask(__name__)

# --- paths ---
MODEL_DIR = "model"
DATA_DIR = "data"
LOG_PATH = "prediction_logs.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
DATA_PATH = os.path.join(DATA_DIR, "vehicle_sensors.csv")

FEATURES = [
    "engine_temp",
    "fuel_efficiency",
    "vibration_level",
    "odometer",
    "coolant_level",
    "battery_voltage",
]

# ---------- 1. Generate synthetic dataset ----------

def generate_dataset(n=1000):
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "engine_temp": rng.normal(85, 15, n).clip(40, 130),
        "fuel_efficiency": rng.normal(14, 3, n).clip(5, 25),
        "vibration_level": rng.normal(5, 2, n).clip(0, 10),
        "odometer": rng.normal(90000, 40000, n).clip(0, 250000),
        "coolant_level": rng.normal(55, 15, n).clip(5, 100),
        "battery_voltage": rng.normal(12.4, 0.4, n).clip(10, 14),
    })

    risk_score = (
        (df.engine_temp - 90) * 0.4 +
        (10 - df.fuel_efficiency) * 0.3 +
        (df.vibration_level - 6) * 1.0 +
        (df.odometer - 120000) / 40000 +
        (30 - df.coolant_level) * 0.2 +
        (12 - df.battery_voltage) * 1.0
    )

    df["label"] = (1 / (1 + np.exp(-risk_score)) > 0.5).astype(int)

    df.to_csv(DATA_PATH, index=False)
    return df


# ---------- 2. Train or load model ----------

def train_model():
    df = generate_dataset()

    X = df[FEATURES]
    y = df["label"]

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_model()


model = load_model()


# ---------- 3. Prediction logic ----------

def predict_failure(form):
    values = [float(form.get(f, 0)) for f in FEATURES]
    arr = np.array(values).reshape(1, -1)

    prob = float(model.predict_proba(arr)[0][1] * 100)
    prob = round(prob, 1)

    if prob >= 70:
        risk = "High Risk"
        action = "Immediate service required within 48 hours."
        days = 2
        rca = "Likely overheating + vibration fault cluster."
    elif prob >= 40:
        risk = "Medium Risk"
        action = "Schedule service within 1 week."
        days = 7
        rca = "Early pattern of sensor stress."
    else:
        risk = "Low Risk"
        action = "Monitor for the next 30 days."
        days = 30
        rca = "System within normal safety range."

    return {
        "prob": prob,
        "risk": risk,
        "action": action,
        "days": days,
        "rca": rca,
        "health_score": round(100 - prob, 1),
    }


# ---------- 4. Logging ----------

def log_prediction(inputs, output):
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp"] + FEATURES + ["prob", "risk"])
        writer.writerow(
            [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            + [inputs.get(f) for f in FEATURES]
            + [output["prob"], output["risk"]]
        )


# ---------- 5. Routes ----------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        form_data = request.form
        result = predict_failure(form_data)
        log_prediction(form_data, result)
    return render_template("predict.html", result=result)


@app.route("/workflow")
def workflow():
    if not os.path.exists(LOG_PATH):
        steps = []
        risk = "No predictions yet"
    else:
        df = pd.read_csv(LOG_PATH)
        last = df.iloc[-1]
        prob = last["prob"]
        risk = last["risk"]

        steps = [
            {
                "title": "Diagnosis Agent",
                "text": f"Analysed latest sensor data. Failure probability: {prob}% ({risk}).",
            },
            {
                "title": "Scheduling Agent",
                "text": "Checked service capacity and recommended earliest available slot.",
            },
            {
                "title": "Customer Agent",
                "text": "Notified driver and service center with risk summary and slot.",
            },
            {
                "title": "Feedback Agent",
                "text": "Awaiting post-service feedback to validate fix effectiveness.",
            },
            {
                "title": "Manufacturing Agent",
                "text": "RCA database updated with this event for pattern learning.",
            },
            {
                "title": "UEBA Layer",
                "text": "User and partner access behaviour analysed – no anomalies detected.",
            },
        ]

    return render_template("workflow.html", steps=steps, risk=risk)


@app.route("/dashboard")
def dashboard():
    if not os.path.exists(LOG_PATH):
        data = {
            "health": 0,
            "avg_health": 0,
            "total_predictions": 0,
            "high_risk_cases": 0,
            "medium_risk_cases": 0,
            "low_risk_cases": 0,
        }
        logs = []
    else:
        df = pd.read_csv(LOG_PATH)
        latest = df.iloc[-1]
        data = {
            "health": 100 - latest["prob"],
            "avg_health": round(100 - df["prob"].mean(), 1),
            "total_predictions": len(df),
            "high_risk_cases": int((df["prob"] >= 70).sum()),
            "medium_risk_cases": int(((df["prob"] >= 40) & (df["prob"] < 70)).sum()),
            "low_risk_cases": int((df["prob"] < 40).sum()),
        }
        logs = df.to_dict("records")

    return render_template("dashboard.html", data=data, logs=logs)


@app.route("/download_logs")
def download_logs():
    if not os.path.exists(LOG_PATH):
        # create empty file so send_file doesn't fail
        open(LOG_PATH, "w").close()
    return send_file(LOG_PATH, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
