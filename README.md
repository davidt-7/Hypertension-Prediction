# 🫀 Predictive Pulse — AI-Powered Hypertension Risk Assessment

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=flat-square&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A machine learning web application that predicts hypertension stages in patients based on clinical inputs — built with Logistic Regression, Flask, and a professional medical UI.

---

## 🚨 The Problem

Hypertension affects over **1.3 billion people worldwide** and is often called the **"silent killer"** — most patients don't know they have it until it causes a heart attack or stroke. Early detection is critical, but access to doctors isn't always immediate.

**Predictive Pulse** gives anyone instant preliminary risk assessment based on their health details — no doctor visit required.

---

## 🎯 What It Does

A user fills in a simple form with their clinical details:

- Demographics (Age, Gender)
- Medical history (Family history, current care status, medication)
- Symptoms (Shortness of breath, vision changes, nosebleeds, severity)
- Blood pressure readings (Systolic & Diastolic)
- Lifestyle factors (Diet, time since diagnosis)

The AI model analyzes these inputs and classifies the patient into one of **4 hypertension stages**:

| Stage | Risk Level | Color |
|-------|-----------|-------|
| Normal | ✅ Low Risk | Green |
| Stage 1 Hypertension | ⚠️ Moderate Risk | Amber |
| Stage 2 Hypertension | 🔶 High Risk | Orange |
| Hypertensive Crisis | 🔴 Emergency | Red |

Each result also includes **personalized medical recommendations**.

---

## 🤖 Machine Learning Details

### Dataset
- **Source:** Kaggle
- **Size:** 1,825 patient records → cleaned to **1,348** (removed 477 duplicates)
- **Features:** 13 clinical attributes
- **Target:** Hypertension stage (4 classes)

### Why Logistic Regression?

We tested **7 algorithms**:

| Algorithm | Accuracy | Decision |
|-----------|----------|----------|
| Decision Tree | 100% | ❌ Overfitted |
| Random Forest | 100% | ❌ Overfitted |
| SVM | 100% | ❌ Overfitted |
| KNN | 98.1% | ⚠️ Considered |
| **Logistic Regression** | **95.2%** | ✅ **Selected** |
| Ridge Classifier | 90.0% | ⚠️ Considered |
| Naive Bayes | 84.4% | ⚠️ Considered |

> **Why reject 100% accuracy?** Decision Tree, Random Forest and SVM *memorized* the training data (overfitting). They fail on new, unseen patients. In medical applications, a model that generalizes reliably to new patients is far more valuable than one with a perfect score on known data.

---

## 🏗️ Project Structure

```
HYPERTENSION PREDICTION/
│
├── patient_data.csv          # Raw dataset (1,825 patient records)
├── model_training.py         # ML training script — run this first
├── logreg_model.pkl          # Saved trained model (auto-generated)
├── app.py                    # Flask backend
│
├── static/
│   └── style.css             # Professional medical UI styling
│
└── templates/
    └── index.html            # Frontend webpage
```

---

## ⚙️ How To Run It

### 1. Clone the Repository
```bash
git clone https://github.com/davidt-7/Hypertension-Prediction.git
cd Hypertension-Prediction
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn flask joblib
```

### 3. Train the Model
```bash
python model_training.py
```
This generates `logreg_model.pkl` and saves 3 analysis charts.

### 4. Start the Web App
```bash
python app.py
```

### 5. Open in Browser
```
http://127.0.0.1:5000
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | Scikit-learn (Logistic Regression) |
| Data Processing | Pandas, NumPy |
| Visualizations | Matplotlib, Seaborn |
| Backend | Python + Flask |
| Frontend | HTML5 + CSS3 |
| Model Serialization | Joblib (.pkl) |

---

## 📊 Results

- **Model Accuracy:** 95.2%
- **Macro F1 Score:** 0.95
- **Crisis Recall:** 100% *(the model never misses a hypertensive emergency)*
- **Training Size:** 1,078 patients
- **Test Size:** 270 patients

---

## 🔮 Future Improvements

- [ ] SHAP-based Explainable AI — show *why* the model made its decision
- [ ] Exact blood pressure number inputs instead of ranges
- [ ] Downloadable PDF patient report
- [ ] Doctor vs Patient view modes
- [ ] Integration with Electronic Health Records (EHR)
- [ ] Wearable device API for automatic blood pressure input
- [ ] Clinical validation for real hospital deployment

---

## ⚠️ Disclaimer

This tool is intended for **educational and preliminary screening purposes only**. It is **not a substitute for professional medical diagnosis**. Always consult a qualified healthcare provider for medical decisions.

