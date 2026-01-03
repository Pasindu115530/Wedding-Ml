import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
# Ensure the file exists in the same directory
df = pd.read_excel('Wedding_Dataset_file_final.xlsx')

# -----------------------------
# 2. Define Features and Target
# -----------------------------
target = 'Total_Wedding_Cost_LKR'

categorical_columns = [
    'Wedding_Season', 'Venue_Type', 'Catering_Type',
    'Decoration_Level', 'Photography_Package', 'Entertainment_Type'
]

numeric_columns = [
    'Guest_Count', 'Venue_Cost_LKR', 'Catering_Cost_LKR',
    'Decoration_Cost_LKR', 'Photography_Cost_LKR', 'Entertainment_Cost_LKR'
]

# Order is vital: Categorical features must come first as per app.py logic
feature_order = categorical_columns + numeric_columns

# -----------------------------
# 3. Handle Missing Values & Cleaning
# -----------------------------
# Fill missing numeric values with the median
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Clean categorical text: lowercase and strip whitespace
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0]).astype(str).str.lower().str.strip()

# -----------------------------
# 4. Save Medians for API (New Requirement)
# -----------------------------
# We save these BEFORE scaling so the API gets real-world numbers
medians = df[numeric_columns].median().to_dict()
joblib.dump(medians, 'medians.pkl') # Saved to be loaded by app.py

# -----------------------------
# 5. Encoding and Scaling
# -----------------------------
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# -----------------------------
# 6. Train Model
# -----------------------------
X = df[feature_order]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using 200 trees for a balance between speed and accuracy
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 7. Save Final Artifacts
# -----------------------------
joblib.dump(model, 'wedding_cost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("--- Training Complete ---")
print(f"Model saved: wedding_cost_model.pkl")
print(f"Medians saved: medians.pkl")
print(f"Calculated Median Guest Count: {medians['Guest_Count']}")