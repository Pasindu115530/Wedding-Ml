import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Set visualization style
sns.set_theme(style="whitegrid")

# 1. Load Data
file_path = "Wedding_Dataset_file_final.xlsx"
try:
    df = pd.read_excel(file_path)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# 2. Data Cleaning (Fixing "same word but change" issue)
# We convert all categorical text to lowercase and strip whitespace
# to ensure "Colombo", "colombo", and " Colombo " are treated as the same.
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    # Convert to string, strip whitespace, modify to title case for consistency
    df[col] = df[col].astype(str).str.strip().str.title()
    print(f"Cleaned column: {col}")

# 3. Handle Missing Values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
# Re-select cat_cols after potential type changes
cat_cols = df.select_dtypes(include=['object']).columns 
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

# 4. Encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 5. Feature Scaling
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 6. Feature Selection (Removing Leaked Columns)
# We deliberately remove component costs because they 'give away' the answer.
cost_columns = [
    'Venue_Cost_LKR', 'Catering_Cost_LKR', 'Decoration_Cost_LKR', 
    'Photography_Cost_LKR', 'Entertainment_Cost_LKR'
]
# Ensure we only drop what exists
existing_cost_cols = [c for c in cost_columns if c in df.columns]

if 'Total_Wedding_Cost_LKR' in df.columns:
    # also drop Wedding_ID as requested
    X = df.drop(['Total_Wedding_Cost_LKR', 'Wedding_ID'] + existing_cost_cols, axis=1, errors='ignore')
    y = df['Total_Wedding_Cost_LKR']
else:
    print("Target column 'Total_Wedding_Cost_LKR' not found.")
    exit(1)

# 7. Train Model (Random Forest is best for Feature Importance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# 8. Calculate Feature Importance
# Feature importance tells us how much each feature contributes to reducing
# the prediction error. Higher values mean the feature is more influential.
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 9. Plot Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
plt.title('Feature Importance (Corrected & Cleaned Data)')
plt.xlabel('Importance Score (Higher is more influential)')
plt.tight_layout()
plt.savefig("feature_importance_cleaned.png")
plt.close()

print("Feature Importance Graph saved as feature_importance_cleaned.png")
print("-" * 30)
print(feature_importance_df)
