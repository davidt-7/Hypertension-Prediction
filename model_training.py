# ============================================================
#   HYPERTENSION PREDICTION - MODEL TRAINING (FIXED VERSION)
#   Run: python model_training.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=============================================")
print("   HYPERTENSION PREDICTION - MODEL TRAINING")
print("=============================================")

# -------------------------------------------------------
# STEP 1: LOAD THE DATASET
# -------------------------------------------------------
print("\n[1/6] Loading dataset...")
data = pd.read_csv('patient_data.csv')
print(f"      Loaded {data.shape[0]} rows and {data.shape[1]} columns")

# Strip whitespace from ALL column names and ALL string values
# This fixes hidden space issues in the CSV
data.columns = data.columns.str.strip()
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

print("      Unique values in each column:")
for col in data.columns:
    print(f"      {col}: {data[col].unique()[:5]}")

# -------------------------------------------------------
# STEP 2: CLEAN THE DATA
# -------------------------------------------------------
print("\n[2/6] Cleaning data...")

# Rename column 'C' to 'Gender' if needed
if 'C' in data.columns:
    data.rename(columns={'C': 'Gender'}, inplace=True)

# Remove duplicates
before = len(data)
data.drop_duplicates(inplace=True)
after = len(data)
print(f"      Removed {before - after} duplicates. {after} records remaining.")

# -------------------------------------------------------
# STEP 3: EDA CHARTS
# -------------------------------------------------------
print("\n[3/6] Creating charts...")

try:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x='Gender', palette='Set2')
    plt.title('Gender Distribution')
    plt.tight_layout()
    plt.savefig('chart_gender.png')
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x='Stages', palette='coolwarm')
    plt.title('Hypertension Stages Distribution')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('chart_stages.png')
    plt.close()

    print("      Charts saved!")
except Exception as e:
    print(f"      Chart warning (non-critical): {e}")

# -------------------------------------------------------
# STEP 4: ENCODE DATA
# We use a safe approach - encode based on actual values found
# -------------------------------------------------------
print("\n[4/6] Encoding data...")

# Make a copy to work on
df = data.copy()

# --- Gender ---
print(f"      Gender values: {df['Gender'].unique()}")
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# --- Binary Yes/No columns ---
binary_cols = ['History', 'Patient', 'TakeMedication',
               'BreathShortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet']
for col in binary_cols:
    if col in df.columns:
        print(f"      {col} values: {df[col].unique()}")
        df[col] = df[col].map({'No': 0, 'Yes': 1})

# --- Age ---
print(f"      Age values: {df['Age'].unique()}")
df['Age'] = df['Age'].map({'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4})

# --- Severity ---
print(f"      Severity values: {df['Severity'].unique()}")
df['Severity'] = df['Severity'].map({'Mild': 0, 'Moderate': 1, 'Severe': 2})

# --- When Diagnosed ---
print(f"      Whendiagnoused values: {df['Whendiagnoused'].unique()}")
df['Whendiagnoused'] = df['Whendiagnoused'].map({
    '<1 Year': 1, '1 - 5 Years': 2, '>5 Years': 3
})

# --- Systolic ---
print(f"      Systolic values: {df['Systolic'].unique()}")
df['Systolic'] = df['Systolic'].map({
    '100 - 110': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3
})

# --- Diastolic ---
print(f"      Diastolic values: {df['Diastolic'].unique()}")
df['Diastolic'] = df['Diastolic'].map({
    '70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3
})

# --- Stages (Target) ---
print(f"      Stages values: {df['Stages'].unique()}")
df['Stages'] = df['Stages'].map({
    'NORMAL': 0,
    'HYPERTENSION (Stage-1)': 1,
    'HYPERTENSION (Stage-2)': 2,
    'HYPERTENSIVE CRISIS': 3
})

# Check how many rows survived encoding
print(f"\n      Rows before dropna: {len(df)}")
df.dropna(inplace=True)
print(f"      Rows after dropna: {len(df)}")
print(f"      Stage distribution:\n{df['Stages'].value_counts()}")

if len(df) < 50:
    print("\n      ERROR: Too many rows dropped!")
    print("      This means the CSV values don't match expected strings.")
    print("      Please share what you see above and I will fix it.")
    exit()

# -------------------------------------------------------
# STEP 5: TRAIN THE MODEL
# -------------------------------------------------------
print("\n[5/6] Training the model...")

X = df.drop('Stages', axis=1)
y = df['Stages']

# Scale ordinal features to 0-1 range
X = X.copy()
X['Age']            = (X['Age'] - 1) / 3
X['Severity']       = X['Severity'] / 2
X['Whendiagnoused'] = (X['Whendiagnoused'] - 1) / 2
X['Systolic']       = X['Systolic'] / 3
X['Diastolic']      = X['Diastolic'] / 3

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# Train!
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("      Model trained successfully!")

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n=============================================")
print("   RESULTS")
print("=============================================")
print(f"   Accuracy: {acc * 100:.2f}%")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Normal', 'Stage-1', 'Stage-2', 'Crisis']))
print("   Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------------
# STEP 6: SAVE THE MODEL
# -------------------------------------------------------
print("\n[6/6] Saving model...")
joblib.dump(model, 'logreg_model.pkl')
print("      Saved as logreg_model.pkl")
print("\n=============================================")
print("   TRAINING COMPLETE!")
print("   Next step: run app.py")
print("=============================================\n")
