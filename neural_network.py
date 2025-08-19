import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set seaborn style for plots
sns.set(style='whitegrid')

# Load dataset
df_path = "data/cleaned_leeds_data2.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(f"❌ File not found: {df_path}")

df = pd.read_csv(df_path)
print("📄 Data loaded. Shape:", df.shape)

# Create log-transformed target if not already present
if "avg_price_log" not in df.columns and "avg_price" in df.columns:
    df = df[df["avg_price"] > 0]
    df["avg_price_log"] = np.log1p(df["avg_price"])
    print("✅ Created 'avg_price_log' column")

# Drop rows with missing values
df.dropna(inplace=True)

# Drop known misleading volume-based features
exclude_cols = [
    "avg_price", "avg_price_log",  # target and raw target
    "sales_volume", "cash_sales", "mortgage_sales", "cash_sales_log"  # misleading predictors
]

# Select numeric features excluding those above
X = df.select_dtypes(include=[np.number]).drop(columns=exclude_cols, errors="ignore")
y = df["avg_price_log"]

if X.empty:
    raise ValueError("❌ No numeric features found for training after exclusion.")

print("📋 Features used for training:", X.columns.tolist())

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("📊 Train shape:", X_train.shape, "| Test shape:", X_test.shape)

# Train MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000,
                   early_stopping=True, random_state=42)
mlp.fit(X_train, y_train)

# Predict and inverse-transform
y_pred_log = mlp.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)

# Evaluate performance
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print("\n📈 MLP Regressor Performance:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  R²:   {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_actual, y=y_pred, alpha=0.7)
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
plt.xlabel("Actual Price (£)")
plt.ylabel("Predicted Price (£)")
plt.title("MLP Regressor – Actual vs Predicted")
plt.tight_layout()

# Save plot
os.makedirs("results", exist_ok=True)
plt.savefig("results/mlp_predictions.png")
plt.show()

# Save predictions
predictions_df = pd.DataFrame({"Actual": y_actual, "Predicted": y_pred})
predictions_df.to_csv("results/mlp_prediction.csv", index=False)
print("📤 MLP predictions exported to results/mlp_prediction.csv")
