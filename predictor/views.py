import os
import pickle
import numpy as np
import pandas as pd
from django.shortcuts import render

# Load model, scaler, and imputer once at startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'predictor/models/random_forest_model_colab.pkl'), 'rb') as file:
    rf_model = pickle.load(file)

with open(os.path.join(BASE_DIR, 'predictor/models/scaler_colab.pkl'), 'rb') as file:
    scaler = pickle.load(file)

with open(os.path.join(BASE_DIR, 'predictor/models/imputer_colab.pkl'), 'rb') as file:
    imputer = pickle.load(file)

# Feature order (must match training data)
feature_order = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang',
                 'oldpeak', 'ca', 'cp_1', 'cp_2', 'cp_3', 'restecg_1',
                 'restecg_2', 'slope_1', 'slope_2', 'thal_1', 'thal_2', 'thal_3']

def predict(request):
    if request.method == "POST":
        # Collect user input from form data
        data = request.POST
        
        try:
            # Parse input values into the correct format
            user_input = [0] * len(feature_order)
            user_input[0] = float(data['age'])
            user_input[1] = float(data['sex'])
            user_input[2] = float(data['trestbps'])
            user_input[3] = float(data['chol'])
            user_input[4] = float(data['fbs'])
            user_input[5] = float(data['thalach'])
            user_input[6] = float(data['exang'])
            user_input[7] = float(data['oldpeak'])
            user_input[8] = float(data['ca'])

            # One-hot encoding for categorical features (cp, restecg, slope, thal)
            cp_value = int(data['cp'])
            if cp_value >= 1:
                user_input[8 + cp_value] = 1

            restecg_value = int(data['restecg'])
            if restecg_value >= 1:
                user_input[11 + restecg_value] = 1

            slope_value = int(data['slope'])
            if slope_value >= 1:
                user_input[14 + slope_value] = 1

            thal_value = int(data['thal'])
            if thal_value >= 1:
                user_input[16 + thal_value] = 1

            # Convert to DataFrame and process with imputer and scaler
            input_df = pd.DataFrame([user_input], columns=feature_order)
            input_imputed = imputer.transform(input_df)
            input_scaled = scaler.transform(input_imputed)

            # Make prediction and calculate probability
            prediction = rf_model.predict(input_scaled)[0]
            probability = rf_model.predict_proba(input_scaled)[:, 1][0]

            # Determine risk level based on probability
            if probability >= 0.55:
                risk_level = "High Risk"
            elif probability >= 0.40:
                risk_level = "Moderate Risk"
            elif probability >= 0.20:
                risk_level = "Low Risk"
            else:
                risk_level = "Very Low Risk"

        except Exception as e:
            return render(request, "predict.html", {"error": f"Error: {e}"})

        # Render results in the template
        return render(request, "predict.html", {
            "prediction": "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected",
            "probability": round(probability * 100, 2),
            "risk_level": risk_level,
        })

    return render(request, "predict.html")
