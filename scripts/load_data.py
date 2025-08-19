import pandas as pd

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads and cleans the Leeds house price dataset from CSV.
    Automatically drops extra columns and assigns 29 proper names.
    """
    try:
        # Load raw file, skipping first 2 rows
        df = pd.read_csv(filepath, encoding='utf-8-sig', header=None, skiprows=2)

        # ✅ Drop last 2 columns (assumed junk)
        df = df.iloc[:, :29]

        # ✅ Assign 29 correct column names
        df.columns = [
            "region", "month_id", "avg_price", "property_type", "sales_volume",
            "price_change", "annual_change", "detached_price", "detached_change", "detached_annual",
            "semi_price", "semi_change", "semi_annual",
            "terraced_price", "terraced_change", "terraced_annual",
            "flat_price", "flat_change", "flat_annual",
            "cash_price", "cash_change", "cash_annual",
            "cash_sales", "mortgage_price", "mortgage_change", "mortgage_annual",
            "mortgage_sales", "date", "interest_rate"
        ]

        # Type conversions
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['avg_price'] = pd.to_numeric(df['avg_price'], errors='coerce')

        print("✅ Data loaded and cleaned successfully.")
        print("📊 Final columns:", df.columns.tolist())
        return df

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

