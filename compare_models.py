import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set directory for results
results_dir = "results"

# Map model names to their prediction files
model_files = {
    "Random Forest": "random_forest_predictions.csv",
    "MLP Regressor": "mlp_prediction.csv",  # corrected filename if needed
   # "TensorFlow 1": "tensorflow_model_1_predictions.csv",
    "TensorFlow 2": "tensorflow_model_2_predictions.csv",
    "TensorFlow 3": "tensorflow_model_3_predictions.csv",
    "Hybrid Model": "hybrid_predictions.csv"
}

# For storing performance metrics
metrics = []

# Plot layout
plt.figure(figsize=(18, 10))
plot_idx = 1

# Evaluate and visualize each model
for model_name, file_name in model_files.items():
    file_path = os.path.join(results_dir, file_name)

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_name}")
        continue

    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Validate required columns
    if "actual" not in df.columns or "predicted" not in df.columns:
        print(f"⚠️ Missing 'actual' or 'predicted' columns in {file_name}. Skipping...")
        continue

    y_true = df["actual"]
    y_pred = df["predicted"]

    # Compute evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics.append({
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    })

    # Scatter plot
    plt.subplot(2, 3, plot_idx)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(model_name)
    plot_idx += 1

# Finalize and save plot
plt.tight_layout()
plt.suptitle("📊 Model Predictions vs Actual", fontsize=16, y=1.02)
plt.savefig(os.path.join(results_dir, "model_predictions_vs_actual.png"))
plt.show()

# Create metrics summary
metrics_df = pd.DataFrame(metrics).sort_values(by="RMSE")

# Display and save
print("\n📋 Model Performance Comparison:")
print(metrics_df.to_string(index=False))
metrics_df.to_csv(os.path.join(results_dir, "model_comparison_metrics.csv"), index=False)
