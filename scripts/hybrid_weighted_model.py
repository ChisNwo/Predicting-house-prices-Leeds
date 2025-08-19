import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define project and results directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "..", "results")

# Load predictions from CSV files
rf_df = pd.read_csv(os.path.join(RESULTS_DIR, "random_forest_predictions.csv"))
mlp_df = pd.read_csv(os.path.join(RESULTS_DIR, "mlp_predictions.csv"))
tf_df = pd.read_csv(os.path.join(RESULTS_DIR, "tensorflow_model_3_predictions.csv"))

# Sanity check
assert rf_df.shape == mlp_df.shape == tf_df.shape, "Mismatch in prediction sizes"

# Extract actuals (should be the same in all files)
actual = rf_df["Actual"]

# Weighted average: give more weight to better models (e.g., Random Forest)
rf_weight = 0.5
mlp_weight = 0.3
tf_weight = 0.2

hybrid_pred = (
    rf_weight * rf_df["Predicted"] +
    mlp_weight * mlp_df["Predicted"] +
    tf_weight * tf_df["Predicted"]
)

# Evaluation
rmse = np.sqrt(mean_squared_error(actual, hybrid_pred))
mae = mean_absolute_error(actual, hybrid_pred)
r2 = r2_score(actual, hybrid_pred)

print("\n🔗 Weighted Hybrid Model (RF + MLP + TF3):")
print(f"  RMSE: {rmse:,.2f}")
print(f"  MAE:  {mae:,.2f}")
print(f"  R²:   {r2:.4f}")

# Save to CSV
output_df = pd.DataFrame({
    "Actual": actual,
    "Hybrid_Prediction": hybrid_pred
})
output_df.to_csv(os.path.join(RESULTS_DIR, "hybrid_weighted_predictions.csv"), index=False)
print("📤 Saved weighted hybrid predictions to results/hybrid_weighted_predictions.csv")
