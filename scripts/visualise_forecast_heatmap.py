import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_prediction_heatmaps(csv_path="data/forecast_comparison.csv", output_dir="results/heatmaps"):
    """
    Generates heatmaps for Actual, Predicted, and Residual values grouped by Year and Month.

    Parameters:
    - csv_path (str): Relative path to the forecast comparison CSV.
    - output_dir (str): Folder to save heatmap images.
    """

    # Resolve full paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # /scripts
    csv_full_path = os.path.normpath(os.path.join(script_dir, "..", csv_path))
    output_dir_full = os.path.normpath(os.path.join(script_dir, "..", output_dir))

    # Check if forecast file exists
    if not os.path.exists(csv_full_path):
        print(f"❌ File not found: {csv_full_path}")
        fallback_dir = os.path.normpath(os.path.join(script_dir, "..", "data"))
        if os.path.exists(fallback_dir):
            print("📁 Available files in 'data':", os.listdir(fallback_dir))
        else:
            print("❌ 'data' directory not found.")
        print("📂 Current working directory:", os.getcwd())
        print("📁 Contents of script directory:", os.listdir(script_dir))
        return

    # Load and preprocess data
    df = pd.read_csv(csv_full_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Residual'] = df['Actual'] - df['Predicted']

    # Group by Year-Month averages
    grouped = df.groupby(['Year', 'Month'])[['Actual', 'Predicted', 'Residual']].mean().unstack()

    # Ensure output directory exists
    os.makedirs(output_dir_full, exist_ok=True)

    # 🔵 Heatmap: Actual Prices
    plt.figure(figsize=(12, 6))
    sns.heatmap(grouped['Actual'], annot=True, cmap='Blues', fmt=".0f")
    plt.title("📊 Average Actual Prices by Year-Month")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_full, "actual_heatmap.png"))
    plt.close()

    # 🟢 Heatmap: Predicted Prices
    plt.figure(figsize=(12, 6))
    sns.heatmap(grouped['Predicted'], annot=True, cmap='Greens', fmt=".0f")
    plt.title("📈 Average Predicted Prices by Year-Month")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_full, "predicted_heatmap.png"))
    plt.close()

    # 🔴 Heatmap: Residuals
    plt.figure(figsize=(12, 6))
    sns.heatmap(grouped['Residual'], annot=True, cmap='coolwarm', center=0, fmt=".1f")
    plt.title("🔥 Residuals (Actual - Predicted) by Year-Month")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_full, "residual_heatmap.png"))
    plt.close()

    print(f"✅ Heatmaps saved in: {output_dir_full}")


# ✅ Run directly if script is executed
if __name__ == "__main__":
    plot_prediction_heatmaps()
