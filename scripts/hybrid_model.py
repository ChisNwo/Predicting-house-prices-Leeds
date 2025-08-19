import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 📁 Define file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# 📄 Load individual model prediction files
rf_df = pd.read_csv(os.path.join(RESULTS_DIR, "random_forest_predictions.csv"))
mlp_df = pd.read_csv(os.path.join(RESULTS_DIR, "mlp_predictions.csv"))
tf3_df = pd.read_csv(os.path.join(RESULTS_DIR, "tensorflow_model_3_predictions.csv"))

# ✅ Normalize column names just in case
for df in [rf_df, mlp_df, tf3_df]:
    df.columns = df.columns.str.lower()

# ✅ Sanity check: all dataframes must match in length
if not (len(rf_df) == len(mlp_df) == len(tf3_df)):
    raise ValueError("Mismatch in prediction sizes across models.")

# 💡 Blend predictions equally (you could weight them differently if desired)
hybrid_pred = (rf_df["predicted"] + mlp_df["predicted"] + tf3_df["predicted"]) / 3
actual = rf_df["actual"]  # Assuming all have same 'actual' values

# 📊 Evaluate hybrid performance
rmse = np.sqrt(mean_squared_error(actual, hybrid_pred))
mae = mean_absolute_error(actual, hybrid_pred)
r2 = r2_score(actual, hybrid_pred)

print("\n🔀 Hybrid Model (RF + MLP + TF3):")
print(f"  RMSE: {rmse:,.2f}")
print(f"  MAE:  {mae:,.2f}")
print(f"  R²:   {r2:.4f}")

# 💾 Save hybrid predictions to CSV
output_df = pd.DataFrame({
    "Actual": actual,
    "Hybrid_Prediction": hybrid_pred
})
output_df.to_csv(os.path.join(RESULTS_DIR, "hybrid_predictions.csv"), index=False)
print("📤 Saved hybrid predictions to results/hybrid_predictions.csv")
