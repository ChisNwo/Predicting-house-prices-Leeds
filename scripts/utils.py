import numpy as np
import pandas as pd

def assign_property_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a fake 'property_age' column for EDA testing.
    Randomly assigns 'new' or 'old' to simulate build types.
    """
    np.random.seed(42)  # Ensures repeatable results
    df['property_age'] = np.random.choice(['new', 'old'], size=len(df))
    return df

def get_sales_extremes_by_age(df: pd.DataFrame):
    """
    Prints the year with the highest and lowest number of new and old builds sold.
    """
    if 'property_age' not in df.columns:
        print("❌ 'property_age' column is missing.")
        return

    if 'sales_volume' not in df.columns:
        print("❌ 'sales_volume' column is missing.")
        return

    if not pd.api.types.is_numeric_dtype(df['sales_volume']):
        print("⚠️ Converting 'sales_volume' to numeric.")
        df['sales_volume'] = pd.to_numeric(df['sales_volume'], errors='coerce')

    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    else:
        print("❌ 'date' column is missing.")
        return

    grouped = df.groupby(['year', 'property_age'])['sales_volume'].sum().reset_index()

    if grouped.empty:
        print("⚠️ Grouped data is empty. No data to summarise.")
        return

    print("\n📈 Yearly Sales Volume by Age (sample):")
    print(grouped.head())

    for age in ['new', 'old']:
        subset = grouped[grouped['property_age'] == age]

        if subset.empty:
            print(f"⚠️ No data for property_age = '{age}'.")
            continue

        max_row = subset.loc[subset['sales_volume'].idxmax()]
        min_row = subset.loc[subset['sales_volume'].idxmin()]

        print(f"\n🏆 Highest sales for '{age}' builds:")
        print(f"Year: {int(max_row['year'])}, Sales Volume: {int(max_row['sales_volume'])}")

        print(f"🔻 Lowest sales for '{age}' builds:")
        print(f"Year: {int(min_row['year'])}, Sales Volume: {int(min_row['sales_volume'])}")
