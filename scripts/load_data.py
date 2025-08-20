import pandas as pd

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads and cleans the Leeds house price dataset from CSV.
    - Drops junk columns
    - Assigns proper column names
    - Converts date and numeric types
    - Filters dates from 1995 onwards
    - Checks for missing values
    - Outputs summary statistics
    """
    try:
        # Load raw file, skipping first 2 rows
        df = pd.read_csv(filepath, encoding='utf-8-sig', header=None, skiprows=2)

        # ✅ Drop last 2 columns (assumed junk)
        df = df.iloc[:, :29]

        # ✅ Assign correct column names
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

        # ✅ Type conversions
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['avg_price'] = pd.to_numeric(df['avg_price'], errors='coerce')

        # ✅ Filter out rows before 1995
        df = df[df['date'] >= pd.Timestamp('1995-01-01')]

        # ✅ Missing value check
        missing_summary = df.isnull().sum()
        print("🔍 Missing values per column:\n", missing_summary[missing_summary > 0])

        # ✅ Describe data
        print("📈 Summary statistics (numerical features):")
        print(df.describe())

        print(f"✅ Dataset loaded and filtered. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()
