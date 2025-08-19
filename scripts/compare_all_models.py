import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Define project root and results directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.join(root_dir, "results")
latex_path = os.path.join(results_dir, "model_comparison_table.tex")
plot_path = os.path.join(results_dir, "model_comparison_plot.png")
csv_path = os.path.join(results_dir, "model_comparison_metrics.csv")

# All models to compare
models = {
    "Random Forest": "random_forest_predictions.csv",
    "MLP Regressor": "mlp_predictions.csv",
   # "TensorFlow 1": "tensorflow_model_1_predictions.csv",
    "TensorFlow 2": "tensorflow_model_2_predictions.csv",
    "TensorFlow 3": "tensorflow_model_3_predictions.csv",
    "Hybrid Model": "hybrid_predictions.csv",
    "Weighted Hybrid": "hybrid_weighted_predictions.csv",
    "Stacking Regressor": "stacking_predictions.csv",
}

performance = []

# Evaluate models
for name, file in models.items():
    file_path = os.path.join(results_dir, file)
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}. Skipping...")
        continue

    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ["Actual", "Predicted"]):
        print(f"⚠️ Missing 'Actual' or 'Predicted' in {file}. Skipping...")
        continue

    y_true = df["Actual"]
    y_pred = df["Predicted"]

    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        performance.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2})
    except Exception as e:
        print(f"❌ Error evaluating {name}: {e}")

# Create summary table
performance_df = pd.DataFrame(performance)
if performance_df.empty:
    print("⚠️ No valid model predictions found. Skipping export and plot.")
    exit()

performance_df = performance_df.sort_values(by="RMSE")
performance_df.to_csv(csv_path, index=False)
performance_df.to_latex(latex_path, index=False, float_format="%.2f")
print(f"\n📋 Model Performance Summary:\n{performance_df}")
print(f"\n📄 LaTeX table saved to {latex_path}")

# Plot
plt.figure(figsize=(10, 6))
melted = performance_df.melt(id_vars="Model", value_vars=["RMSE", "MAE", "R²"])
for metric in ["RMSE", "MAE", "R²"]:
    subset = melted[melted["variable"] == metric]
    plt.bar(subset["Model"], subset["value"], label=metric)

plt.title("📊 Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)
print(f"📈 Plot saved to {plot_path}")
