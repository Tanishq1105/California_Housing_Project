from flask import Flask, render_template, request, flash
import numpy as np
import joblib
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model

# ---------------------------------------------------
# App Setup
# ---------------------------------------------------
app = Flask(__name__)
app.secret_key = "super_secret_key"

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------
# Model Configuration
# ---------------------------------------------------
MODEL_PATHS = {
    "regression": "models/regression_model.pkl",
    "rf": "models/classifier_model.pkl",
    "svm": "models/svm_model.pkl",
    "scaler": "models/scaler.pkl"
}

NN_MODEL_PATH = "models/neural_network_model.keras"

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
    logging.error(f"Error loading models: {e}")

# ---------------------------------------------------
# Helper Function
# ---------------------------------------------------
def get_predictions(features):
    """
    Transform input and return predictions
    from all models.
    """

    import pandas as pd

    columns = [
    "MedInc","HouseAge","AveRooms","AveBedrms",
    "Population","AveOccup","Latitude","Longitude"
    ]

    df_input = pd.DataFrame([features], columns=columns)

    data = models["scaler"].transform(df_input)


    # Regression
    reg_pred = models["regression"].predict(data)[0]

    # Classification
    rf_pred = LABELS[models["rf"].predict(data)[0]]
    svm_pred = LABELS[models["svm"].predict(data)[0]]

    # Neural Network
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

        except ValueError:
            flash("Please enter valid numeric values.")
        except KeyError as e:
            flash(f"Missing input field: {e}")
        except Exception as e:
            flash(f"Unexpected error: {e}")

    return render_template("index.html", results=results)

# ---------------------------------------------------
# Run App
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
