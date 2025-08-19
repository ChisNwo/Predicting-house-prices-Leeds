# tensorflow_models.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load cleaned dataset
df = pd.read_csv("data/cleaned_leeds_data2.csv")
print("\n✅ Data loaded. Shape:", df.shape)

# Create log-transformed target if not exists
if "avg_price_log" not in df.columns and "avg_price" in df.columns:
    df["avg_price_log"] = np.log1p(df["avg_price"])
    print("✅ Created 'avg_price_log' column")

# Drop missing values
df = df.dropna()

# Prepare features and target
X = df.select_dtypes(include=[np.number]).drop(columns=["avg_price", "avg_price_log"], errors="ignore")
y = df["avg_price_log"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to evaluate model
def evaluate_model(y_true, y_pred_log, model_name):
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_true)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    print(f"\n📊 {model_name} Performance:")
    print(f"  RMSE: {rmse:.2f}\n  MAE: {mae:.2f}\n  R²: {r2:.4f}")

    # Save predictions
    predictions_df = pd.DataFrame({"Actual": y_actual, "Predicted": y_pred})
    os.makedirs("results", exist_ok=True)
    predictions_df.to_csv(f"results/{model_name.lower().replace(' ', '_')}_predictions.csv", index=False)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.6)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{model_name} – Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(f"results/{model_name.lower().replace(' ', '_')}_plot.png")
    plt.close()

# ---------------------------
# 🔹 Model 1: Simple MLP
# ---------------------------
#model1 = Sequential([
   # Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # Dense(1)
# ])
# model1.compile(optimizer='adam', loss='mse')
# model1.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
# y_pred1 = model1.predict(X_test).flatten()
# evaluate_model(y_test, y_pred1, "TensorFlow Model 1")

# ---------------------------
# 🔹 Model 2: Deeper Network
# ---------------------------
model2 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
model2.compile(optimizer='adam', loss='mse')
model2.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
y_pred2 = model2.predict(X_test).flatten()
evaluate_model(y_test, y_pred2, "TensorFlow Model 2")

# ---------------------------
# 🔹 Model 3: With Dropout Regularisation
# ---------------------------
model3 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
model3.compile(optimizer='adam', loss='mse')
model3.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
y_pred3 = model3.predict(X_test).flatten()
evaluate_model(y_test, y_pred3, "TensorFlow Model 3")

print("\n✅ All TensorFlow models trained and evaluated. Outputs saved in /results")
