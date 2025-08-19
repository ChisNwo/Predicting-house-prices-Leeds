import os
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(ROOT_DIR, "..", "data", "cleaned_leeds_data2.csv"))
RESULTS_DIR = os.path.normpath(os.path.join(ROOT_DIR, "..", "results"))

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded data: {df.shape}")

# Add log-transformed target if not already present
if "avg_price_log" not in df.columns:
    df["avg_price_log"] = np.log1p(df["avg_price"])

# Define features and target
# Drop target columns
X = df.drop(columns=["avg_price", "avg_price_log"])

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

y = df["avg_price_log"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base learners
base_learners = [
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
    ("svr", SVR(kernel="rbf", C=10, gamma=0.01))
]

# Define final estimator
final_estimator = LinearRegression()

# Define the stacking model
stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=final_estimator,
    passthrough=False,
    n_jobs=-1
)

# Train the model
stacking_model.fit(X_train, y_train)

# Predict and inverse log transform
y_pred_log = stacking_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n📌 Stacking Ensemble Performance:")
print(f"  RMSE: {rmse:,.2f}")
print(f"  MAE:  {mae:,.2f}")
print(f"  R²:   {r2:.4f}")

# Save predictions
pred_df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
pred_path = os.path.join(RESULTS_DIR, "stacking_predictions.csv")
pred_df.to_csv(pred_path, index=False)
print(f"\n📁 Saved predictions to {pred_path}")

# Plot predictions
plt.figure(figsize=(7, 7))
plt.scatter(y_true, y_pred, alpha=0.7)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--", color="red")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("📊 Stacking Regressor: Predicted vs Actual")
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "stacking_predictions_plot.png")
plt.savefig(plot_path)
plt.show()
print(f"📈 Saved plot to {plot_path}")
