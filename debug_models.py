import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    df = pd.read_excel("Wedding_Dataset_file_final.xlsx")
except Exception as e:
    with open("results.txt", "w") as f:
        f.write(f"Error: {e}")
    exit(1)

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop('Total_Wedding_Cost_LKR', axis=1)
y = df['Total_Wedding_Cost_LKR']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

with open("results.txt", "w") as f:
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        f.write(f"{name}: MAE={mae}, RMSE={rmse}\n")
