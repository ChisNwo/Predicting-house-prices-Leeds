# Predicting Leeds House Prices Using Machine Learning

## 📌 Project Overview
This project aims to **predict average house prices in Leeds**, UK, using historical housing and economic data (1995–2024). It applies machine learning techniques such as **Linear Regression**, **Random Forest**, **Support Vector Regression**, and **MLP Regressor** to identify pricing trends, assess model performance, and forecast future prices.

## 📂 Project Structure
```
├── data/
│   ├── raw/                   # Original CSVs
│   └── cleaned/               # Preprocessed datasets
├── scripts/
│   ├── preprocessing.py       # Data cleaning, transformation, and feature engineering
│   ├── train_models.py        # Model training and evaluation
│   ├── forecast.py            # Time series forecasting script
│   └── utils.py               # Helper functions
├── models/
│   └── random_forest_model.joblib  # Saved ML models
├── results/
│   ├── forecast_comparison.csv     # Actual vs predicted prices
│   └── plots/                      # All result visualisations
├── README.md
├── requirements.txt
├── neural_network.py
└── main.py
```

##  How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/ChisNwo/leeds-house-price-prediction.git
cd leeds-house-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the main script
```bash
python main.py
```

## 🧠 Models Used

| Model                  | Purpose                            |
|------------------------|------------------------------------|
| Linear Regression      | Baseline regression model          |
| Random Forest          | Main model with high accuracy      |
| SVR (Support Vector)   | Handles non-linear relationships   |
| MLP Regressor          | Deep learning-based approximation  |

## 🧪 Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Goodness of fit)

## 📊 Visual Outputs

All model evaluations, forecasts, and residuals are saved to:
```
/outputs/plots/
```

Key plots:
- Actual vs Predicted Prices (2021–2025)
- Residual Plots
- Feature Importance
- Forecast Quality Line Charts

## 🛠️ Dependencies

See `requirements.txt` for a full list. Main libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `tensorflow` 

## 📁 Data Sources

- UK House Price Index
- Mortgage Rate Data (BOE)
- Custom CSVs for Leeds from 1995 onwards

## 💾 Model Saving & Loading

Trained models are saved using Joblib:
```python
from joblib import dump, load
dump('model', 'models/random_forest_model.joblib')
model = load('models/random_forest_model.joblib')
```

## 🧪 Testing

Basic unit checks included:
```python
assert df['avg_price_log'].notnull().all()
assert not df.duplicated().any()
```

## 📌 Author

Created by Chis N 
Email: 2411309@leedstrinity.ac.uk
University: Leeds Trinity University.
