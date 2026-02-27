# ============================================================
#   HYPERTENSION PREDICTION - FLASK WEB APP
#   Run: python app.py
#   Then open browser: http://127.0.0.1:5000
# ============================================================

from flask import Flask, render_template, request, flash
import joblib
import numpy as np
import os

# Create the Flask app
# Flask is a lightweight web framework - it connects Python to a webpage
app = Flask(__name__)
app.secret_key = 'hypertension_secret_key'

# -------------------------------------------------------
# LOAD THE TRAINED MODEL
# We saved this in model_training.py - now we load it here
# No need to retrain every time the app starts!
# -------------------------------------------------------
try:
    model = joblib.load('logreg_model.pkl')
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("✗ ERROR: logreg_model.pkl not found! Run model_training.py first.")
    model = None

# -------------------------------------------------------
# STAGE LABELS
# Convert the number the model returns back to a readable label
# 0 = Normal, 1 = Stage-1, 2 = Stage-2, 3 = Crisis
# -------------------------------------------------------
stage_map = {
    0: 'NORMAL',
    1: 'HYPERTENSION (Stage-1)',
    2: 'HYPERTENSION (Stage-2)',
    3: 'HYPERTENSIVE CRISIS'
}

# Color for each stage (shown on the result card)
color_map = {
    0: '#10B981',   # Green  - Normal
    1: '#F59E0B',   # Amber  - Stage 1
    2: '#F97316',   # Orange - Stage 2
    3: '#EF4444'    # Red    - Crisis
}

# Risk label for each stage
risk_map = {
    0: 'LOW RISK',
    1: 'MODERATE RISK',
    2: 'HIGH RISK',
    3: 'EMERGENCY'
}

# Medical recommendations for each stage
recommendations = {
    0: {
        'title': 'Normal Blood Pressure',
        'description': 'Your cardiovascular risk assessment indicates normal blood pressure levels.',
        'actions': [
            'Maintain current healthy lifestyle',
            'Regular physical activity (150 minutes/week)',
            'Continue balanced, low-sodium diet',
            'Annual blood pressure monitoring',
            'Regular health check-ups'
        ]
    },
    1: {
        'title': 'Stage 1 Hypertension',
        'description': 'Mild elevation detected requiring lifestyle modifications and medical consultation.',
        'actions': [
            'Schedule appointment with healthcare provider',
            'Implement DASH diet plan',
            'Increase physical activity gradually',
            'Monitor blood pressure bi-weekly',
            'Reduce sodium intake (<2300mg/day)',
            'Consider stress management techniques'
        ]
    },
    2: {
        'title': 'Stage 2 Hypertension',
        'description': 'Significant hypertension requiring immediate medical intervention and treatment.',
        'actions': [
            'URGENT: Consult physician within 1-2 days',
            'Likely medication therapy required',
            'Comprehensive cardiovascular assessment',
            'Daily blood pressure monitoring',
            'Strict dietary sodium restriction',
            'Lifestyle modification counseling'
        ]
    },
    3: {
        'title': 'Hypertensive Crisis',
        'description': 'CRITICAL: Dangerously elevated blood pressure requiring emergency medical care.',
        'actions': [
            'EMERGENCY: Seek immediate medical attention',
            'Call 911 if experiencing symptoms',
            'Do not delay treatment',
            'Monitor for stroke/heart attack signs',
            'Prepare current medication list',
            'Avoid physical exertion'
        ]
    }
}

# -------------------------------------------------------
# HOME PAGE ROUTE
# When user visits http://127.0.0.1:5000 they see this page
# -------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------------------------------------
# PREDICT ROUTE
# When user clicks "Generate Risk Assessment" button,
# the form data is sent here, we run the model, return result
# -------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all form inputs
        form = request.form

        # -----------------------------------------------
        # ENCODE USER INPUT (same as we did in training!)
        # We must convert text inputs to numbers before
        # feeding them into the model
        # -----------------------------------------------
        gender          = 0 if form['Gender'] == 'Male' else 1
        age             = {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4}[form['Age']]
        history         = 1 if form['History'] == 'Yes' else 0
        patient         = 1 if form['Patient'] == 'Yes' else 0
        take_medication = 1 if form['TakeMedication'] == 'Yes' else 0
        severity        = {'Mild': 0, 'Moderate': 1, 'Severe': 2}[form['Severity']]
        breath          = 1 if form['BreathShortness'] == 'Yes' else 0
        visual          = 1 if form['VisualChanges'] == 'Yes' else 0
        nose            = 1 if form['NoseBleeding'] == 'Yes' else 0
        when_diagnosed  = {'<1 Year': 1, '1 - 5 Years': 2, '>5 Years': 3}[form['Whendiagnoused']]
        systolic        = {'100 - 110': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3}[form['Systolic']]
        diastolic       = {'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3}[form['Diastolic']]
        diet            = 1 if form['ControlledDiet'] == 'Yes' else 0

        # -----------------------------------------------
        # SCALE ordinal features (same as training!)
        # -----------------------------------------------
        age_scaled            = (age - 1) / 3
        severity_scaled       = severity / 2
        when_diagnosed_scaled = (when_diagnosed - 1) / 2
        systolic_scaled       = systolic / 3
        diastolic_scaled      = diastolic / 3

        # -----------------------------------------------
        # BUILD INPUT ARRAY
        # The model expects inputs in the EXACT same order
        # as the training columns
        # -----------------------------------------------
        input_array = np.array([[
            gender,
            age_scaled,
            history,
            patient,
            take_medication,
            severity_scaled,
            breath,
            visual,
            nose,
            when_diagnosed_scaled,
            systolic_scaled,
            diastolic_scaled,
            diet
        ]])

        # -----------------------------------------------
        # MAKE PREDICTION
        # model.predict returns the stage number (0,1,2,3)
        # model.predict_proba returns confidence percentage
        # -----------------------------------------------
        prediction  = model.predict(input_array)[0]
        confidence  = round(max(model.predict_proba(input_array)[0]) * 100, 1)

        # Get the result details
        result_text           = stage_map[prediction]
        result_color          = color_map[prediction]
        result_risk           = risk_map[prediction]
        result_recommendation = recommendations[prediction]

        return render_template('index.html',
                               prediction_text=result_text,
                               result_color=result_color,
                               result_risk=result_risk,
                               confidence=confidence,
                               recommendation=result_recommendation,
                               form_data=form)

    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return render_template('index.html')


# -------------------------------------------------------
# RUN THE APP
# debug=True means it auto-refreshes when you save changes
# -------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
