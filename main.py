# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings and set seaborn style
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

# Project modules
from scripts.load_data import load_dataset
from scripts.eda import (
    plot_property_types_over_time, plot_yearly_sales_volume_by_age, plot_interest_vs_sales,
    plot_correlation_heatmap, get_top_correlations, drop_highly_correlated,
    plot_numeric_distributions, identify_skewed_features, log_transform_selected,
    plot_distribution_and_boxplot
)
from scripts.utils import assign_property_age, get_sales_extremes_by_age
from scripts.model import (
    train_and_evaluate_models, plot_feature_importance,
    tune_random_forest, cross_validate_random_forest,
    forecast_future_prices, export_predictions
)
from scripts.visuals import plot_forecast_vs_actual, plot_forecast_residuals, plot_feature_importance_filtered

def main():
    file_path = "data/UKHPI_Leeds_dataset.csv"
    print("\n🚀 Starting Leeds House Price Prediction Pipeline")

    # Load dataset
    df = load_dataset(file_path)

    print(df.head())

    if df.empty:
        print("❌ Dataset failed to load or is empty.")
        return

    # Fix column mislabeling
    df.rename(columns={
        'avg_price': 'sales_volume',
        'sales_volume': 'avg_price'
    }, inplace=True)

    print(df[['avg_price', 'sales_volume']].head(10))

    # Assign property age
    df = assign_property_age(df)

    # Clean and transform target variable
    df = df[df['avg_price'].notna() & (df['avg_price'] > 0)]
    df['avg_price_log'] = np.log1p(df['avg_price'])

    # EDA & preprocessing
    plot_correlation_heatmap(df)
    plot_numeric_distributions(df)

    skewed_cols = identify_skewed_features(df)
    plot_distribution_and_boxplot(df, skewed_cols)
    df = log_transform_selected(df)

    df.to_csv("data/cleaned_leeds_data2.csv", index=False)
    print("💾 Cleaned dataset saved.")
    print("📊 Final columns:", list(df.columns))

    # EDA visualizations
    plot_property_types_over_time(df)
    plot_yearly_sales_volume_by_age(df)
    plot_interest_vs_sales(df)
    get_top_correlations(df)

    # Drop highly correlated features
    df = drop_highly_correlated(df, threshold=0.95, exclude=['avg_price', 'avg_price_log'])

    # Yearly extremes by property age
    df['year'] = pd.to_datetime(df['date']).dt.year
    grouped = df.groupby(['year', 'property_age'])['sales_volume'].sum().reset_index()

    for age in ['new', 'old']:
        subset = grouped[grouped['property_age'] == age]
        if not subset.empty:
            max_row = subset.loc[subset['sales_volume'].idxmax()]
            min_row = subset.loc[subset['sales_volume'].idxmin()]
            print(f"\n🏆 Highest sales for '{age}' builds: {int(max_row['year'])} – {int(max_row['sales_volume'])}")
            print(f"🔻 Lowest sales for '{age}' builds: {int(min_row['year'])} – {int(min_row['sales_volume'])}")

    # Model training & evaluation
    best_rf, feature_names = train_and_evaluate_models(df, target='avg_price_log')

    if best_rf:
        plot_feature_importance(best_rf, feature_names)
        plot_feature_importance_filtered(
            best_rf, feature_names,
            threshold=0.01,
            save_path="results/filtered_feature_importance.png"
        )

    # Cross-validation
    cv_results = cross_validate_random_forest(df, target='avg_price_log', n_splits=5)
    print("\n📊 Cross-Validation Results:")
    print(cv_results)

    # Hyperparameter tuning
    X = df.select_dtypes(include=[np.number]).drop(columns=['avg_price_log'], errors='ignore')
    y = df['avg_price_log']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_rf = tune_random_forest(X_train, y_train)

    # Final evaluation
    y_pred = np.expm1(best_rf.predict(X_test))
    y_actual = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    print("\n🔧 Tuned Random Forest:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")

    export_predictions(y_actual, y_pred, model_name="Random Forest", filename="results/random_forest_predictions.csv")

    # Forecast future prices
    forecast_future_prices(df)
    plot_forecast_vs_actual()
    plot_forecast_residuals()

if __name__ == "__main__":
    main()
