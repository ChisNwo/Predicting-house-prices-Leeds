import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set(style="whitegrid")


def plot_forecast_vs_actual(csv_path="forecast_comparison.csv"):
    """
    Plot forecasted vs actual house prices over time.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Forecast file not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty or "Date" not in df.columns:
        print("❌ Forecast data is empty or malformed.")
        return

    df['Date'] = pd.to_datetime(df['Date'])

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Actual'], label="Actual", marker='o')
    plt.plot(df['Date'], df['Predicted'], label="Predicted", marker='x')
    plt.title("📈 Forecasted vs Actual House Prices (Post-2020)")
    plt.xlabel("Date")
    plt.ylabel("Average Price (£)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/forecast_vs_actual.png")
    plt.show()


def plot_forecast_residuals(csv_path="forecast_comparison.csv"):
    """
    Plot residuals (actual - predicted) over time to assess prediction error.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Forecast file not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty or "Date" not in df.columns:
        print("❌ Forecast data is empty or malformed.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df['Residual'] = df['Actual'] - df['Predicted']

    mae = np.abs(df['Residual']).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Residual'], label="Residual", color='red')
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(f"🔍 Forecast Residuals Over Time (MAE: £{mae:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Residual (£)")
    plt.tight_layout()
    plt.savefig("results/forecast_residuals.png")
    plt.show()


def plot_feature_importance_filtered(model, feature_names, threshold=0.01, save_path=None):
    """
    Plot feature importances for features exceeding a given threshold.
    """
    if not hasattr(model, "feature_importances_"):
        print("❌ Model does not support feature_importances_.")
        return

    feat_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })

    feat_importance = feat_importance[feat_importance['Importance'] > threshold]
    feat_importance = feat_importance.sort_values(by='Importance', ascending=False)

    if feat_importance.empty:
        print("⚠️ No features exceeded importance threshold.")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feat_importance, x='Feature', y='Importance', palette='Set2')
    plt.title(f"🔍 Feature Importance (Threshold > {threshold})", fontsize=16)
    plt.xticks(rotation=45)
    plt.ylabel("Importance Score")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Feature importance plot saved to {save_path}")
    else:
        plt.show()

