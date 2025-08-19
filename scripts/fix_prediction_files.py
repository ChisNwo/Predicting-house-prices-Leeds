import pandas as pd
import os

# Set your working directory
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, '..', 'results')

# File renaming map: {source_name: (actual_col, predicted_col)}
files_to_fix = {
    "hybrid_predictions.csv": ("Actual", "Hybrid_Prediction"),
    "hybrid_weighted_predictions.csv": ("Actual", "Hybrid_Prediction"),
    "stacking_predictions.csv": ("actual", "predicted")
}

for file, (actual_col, pred_col) in files_to_fix.items():
    file_path = os.path.join(results_dir, file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if actual_col in df.columns and pred_col in df.columns:
            df.rename(columns={actual_col: "Actual", pred_col: "Predicted"}, inplace=True)
            df.to_csv(file_path, index=False)
            print(f"✅ Fixed: {file}")
        else:
            print(f"⚠️ Columns missing in {file}: {df.columns.tolist()}")
    else:
        print(f"❌ File not found: {file_path}")
