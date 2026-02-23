import os
import logging
import numpy as np
import joblib
import pandas as pd
from flask import Flask, render_template, request, flash
from tensorflow.keras.models import load_model

# ---------------------------------------------------
# App Setup
# ---------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_secret")

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "regression": os.path.join(BASE_DIR, "models/regression_model.pkl"),
    "rf": os.path.join(BASE_DIR, "models/classifier_model.pkl"),
    "svm": os.path.join(BASE_DIR, "models/svm_model.pkl"),
    "scaler": os.path.join(BASE_DIR, "models/scaler.pkl")
}

NN_MODEL_PATH = os.path.join(BASE_DIR, "models/neural_network_model.keras")

LABELS = {0: "Low", 1: "Medium", 2: "High"}

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
models = {}
nn_model = None

try:
    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    nn_model = load_model(NN_MODEL_PATH)
    logging.info("All models loaded successfully.")
except Exception as e:
    logging.error(f"Model loading error: {e}")

# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------
def get_predictions(features):

    columns = [
        "MedInc","HouseAge","AveRooms","AveBedrms",
        "Population","AveOccup","Latitude","Longitude"
    ]

    df_input = pd.DataFrame([features], columns=columns)

    data = models["scaler"].transform(df_input)

    reg_pred = models["regression"].predict(data)[0]
    rf_pred = LABELS[models["rf"].predict(data)[0]]
    svm_pred = LABELS[models["svm"].predict(data)[0]]

    nn_probs = nn_model.predict(data, verbose=0)
    nn_pred = LABELS[np.argmax(nn_probs)]

    return {
        "regression": f"{reg_pred:.3f}",
        "rf": rf_pred,
        "svm": svm_pred,
        "nn": nn_pred
    }

# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    results = None

    if request.method == "POST":
        try:
            feature_keys = [
                "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude"
            ]

            features = [float(request.form[key]) for key in feature_keys]

            results = get_predictions(features)

        except Exception as e:
            logging.error(e)
            flash("Something went wrong. Check inputs or server logs.")

    return render_template("index.html", results=results)

# ---------------------------------------------------
# Run (for local only)
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)