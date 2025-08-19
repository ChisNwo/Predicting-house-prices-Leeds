# Importing required libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

# Set seaborn style for cleaner visualizations
sns.set(style='whitegrid')

# Train and evaluate multiple regression models
def train_and_evaluate_models(df: pd.DataFrame, target: str):
    print(f"\n🚀 Running model training for target: {target}")

    if target not in df.columns:
        print(f"❌ Target '{target}' not found.")
        return None, None

    X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Training set: {X_train.shape} | Test set: {X_test.shape}")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Support Vector Regression": SVR(),
        "MLP Regressor": MLPRegressor(random_state=42, max_iter=1000)
    }

    best_model = None
    best_r2 = -np.inf

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_actual = np.expm1(y_test) if 'log' in target else y_test
            y_pred = np.expm1(y_pred) if 'log' in target else y_pred

            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            mae = mean_absolute_error(y_actual, y_pred)
            r2 = r2_score(y_actual, y_pred)

            print(f"\n📈 Results for {name}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE:  {mae:.2f}")
            print(f"  R²:   {r2:.4f}")

            plot_predictions(y_actual, y_pred, name)
            export_predictions(y_actual, y_pred, name, f"results/{name.lower().replace(' ', '_')}_predictions.csv")

            if r2 > best_r2:
                best_model = model
                best_r2 = r2
                best_model_name = name
        except Exception as e:
            print(f"❌ Error with {name}: {e}")

    if best_model:
        save_model(best_model, f"results/{best_model_name.lower().replace(' ', '_')}_model.joblib")
        return best_model, X.columns

    return None, None

# Forecast future house prices using a trained Random Forest model
def forecast_future_prices(df: pd.DataFrame, cutoff_date="2020-12-01", target="avg_price_log"):
    print("\n📈 Forecasting post-cutoff house prices...")
    if target not in df.columns or 'date' not in df.columns:
        print("❌ Required columns missing.")
        return

    train_df = df[df['date'] <= cutoff_date]
    test_df = df[df['date'] > cutoff_date]

    if train_df.empty or test_df.empty:
        print("⚠️ Insufficient data for forecasting.")
        return

    train_df = train_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    X_train = train_df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y_train = train_df[target]

    X_test = test_df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y_test = test_df[target]

    if X_test.empty:
        print("❌ X_test is empty after filtering. Check your data.")
        return

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)

    # ✅ Inverse log transform only if applicable
    if 'log' in target:
        y_pred = np.expm1(y_pred_log)
        actual = np.expm1(y_test)
    else:
        y_pred = y_pred_log
        actual = y_test

    results = pd.DataFrame({
        'Date': test_df['date'],
        'Actual': actual,
        'Predicted': y_pred
    })

    results.to_csv("forecast_comparison.csv", index=False)
    print("📤 Forecast saved to forecast_comparison.csv")

# Tune Random Forest model with grid search
def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='r2'
    )
    grid.fit(X_train, y_train)
    print("\n🔧 Best Random Forest Parameters:")
    print(grid.best_params_)
    return grid.best_estimator_

# Cross-validation for Random Forest model
def cross_validate_random_forest(df: pd.DataFrame, target: str, n_splits=5):
    if target not in df.columns:
        print(f"❌ Target '{target}' not found.")
        return pd.DataFrame()

    X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = df[target]

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    scoring = {
        'RMSE': make_scorer(rmse),
        'MAE': make_scorer(mean_absolute_error),
        'R2': make_scorer(r2_score)
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(RandomForestRegressor(random_state=42), X, y, scoring=scoring, cv=kf)

    summary = {
        'Metric': ['RMSE', 'MAE', 'R²'],
        'Mean': [np.mean(scores['test_RMSE']), np.mean(scores['test_MAE']), np.mean(scores['test_R2'])],
        'STD': [np.std(scores['test_RMSE']), np.std(scores['test_MAE']), np.std(scores['test_R2'])]
    }

    return pd.DataFrame(summary)

# Visualize actual vs predicted values
def plot_predictions(y_actual, y_pred, model_name: str):
    plt.figure(figsize=(8, 6))
    # Scatter plot for actual vs predicted
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.6, label='Predicted Values')

    # Line for perfect prediction (y = x)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', label='Ideal Prediction Line')

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{model_name} – Actual vs Predicted")
    plt.legend()  # ✅ Add legend here
    plt.tight_layout()
    plt.savefig(f"results/{model_name.lower().replace(' ', '_')}_scatterplot.png")
    plt.show()


# Plot top N most important features from Random Forest
def plot_feature_importance(model, feature_names, top_n=10):
    if not hasattr(model, "feature_importances_"):
        print("❌ Model has no feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
    plt.title("🔍 Top Feature Importances (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# Save the trained model to disk
def save_model(model, filename='model.joblib'):
    joblib.dump(model, filename)
    print(f"💾 Model saved to {filename}")

# Save actual vs predicted results to CSV
def export_predictions(y_actual, y_pred, model_name, filename="results/predictions.csv"):
    results = pd.DataFrame({
        "Actual": y_actual,
        "Predicted": y_pred
    })
    results.to_csv(filename, index=False)
    print(f"📤 Predictions exported to {filename} ({model_name})")
