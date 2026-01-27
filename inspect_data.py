import pandas as pd

# Load data
file_path = "Wedding_Dataset_file_final.xlsx"
try:
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Get categorical columns
cat_cols = df.select_dtypes(include=['object']).columns

print("Unique values in categorical columns:")
print("-" * 40)
for col in cat_cols:
    unique_vals = df[col].unique()
    print(f"\nColumn: {col} ({len(unique_vals)} unique values)")
    print(sorted([str(x) for x in unique_vals]))
