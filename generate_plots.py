import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style
sns.set_theme(style="whitegrid")

# Load data
file_path = "Wedding_Dataset_file_final.xlsx"
try:
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Preprocessing
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Feature scaling
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# FIXED: Drop individual cost columns to prevent data leakage
cost_columns = [
    'Venue_Cost_LKR', 'Catering_Cost_LKR', 'Decoration_Cost_LKR', 
    'Photography_Cost_LKR', 'Entertainment_Cost_LKR'
]
existing_cost_cols = [c for c in cost_columns if c in df.columns]

if 'Total_Wedding_Cost_LKR' in df.columns:
    X = df.drop(['Total_Wedding_Cost_LKR'] + existing_cost_cols, axis=1)
    y = df['Total_Wedding_Cost_LKR']
else:
    print("Target column 'Total_Wedding_Cost_LKR' not found.")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)

# 1. Error Metrics Plot (MAE & RMSE)
plt.figure(figsize=(10, 6))
results_df.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", rot=0, ax=plt.gca())
plt.title("Model Error Comparison (Lower is Better)")
plt.ylabel("Error")
plt.tight_layout()
plt.savefig("model_error_comparison.png")
plt.close()

# 2. R2 Score Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="R2", data=results_df, palette="viridis")
plt.title("R2 Score Comparison (Higher is Better)")
plt.ylabel("R2 Score")
# plt.ylim(0, 1)  # Removed to allow negative R2 scores to show
plt.axhline(0, color='black', linewidth=0.8, linestyle='--') # Add reference line at 0
plt.tight_layout()
plt.savefig("model_r2_comparison.png")
plt.close()

print("Graphs generated: model_error_comparison.png, model_r2_comparison.png")
print(results_df)
