import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# 1. AVERAGE PRICE TREND BY PROPERTY TYPE
def plot_property_types_over_time(df: pd.DataFrame):
    df = df.sort_values('date').copy()
    plt.figure(figsize=(14, 6))

    for col, label in zip(
        ['detached_price', 'semi_price', 'terraced_price', 'flat_price'],
        ['Detached', 'Semi-Detached', 'Terraced', 'Flat/Maisonette']
    ):
        if col in df.columns:
            sns.lineplot(
                x='date',
                y=pd.to_numeric(df[col], errors='coerce'),
                data=df,
                label=label,
                linewidth=2
            )

    plt.title("Average House Price by Property Type in Leeds", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("Price (£)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 2. SALES VOLUME BY PROPERTY AGE
def plot_yearly_sales_volume_by_age(df: pd.DataFrame):
    if 'property_age' not in df.columns:
        print("⚠️ 'property_age' column missing. Skipping plot.")
        return

    df['year'] = df['date'].dt.year
    grouped = df.groupby(['year', 'property_age'])['sales_volume'].sum().reset_index()

    plt.figure(figsize=(14, 6))
    sns.barplot(x='year', y='sales_volume', hue='property_age', data=grouped)
    plt.title("Yearly Sales Volume by Property Age")
    plt.ylabel("Total Sales Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 3. INTEREST RATE VS SALES VOLUME
def plot_interest_vs_sales(df: pd.DataFrame):
    df['year'] = df['date'].dt.year
    yearly = df.groupby('year').agg({
        'interest_rate': 'mean',
        'sales_volume': 'sum'
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=yearly, x='year', y='sales_volume', ax=ax1, color='blue')
    ax1.set_ylabel("Sales Volume", color='blue')

    ax2 = ax1.twinx()
    sns.lineplot(data=yearly, x='year', y='interest_rate', ax=ax2, color='red')
    ax2.set_ylabel("Interest Rate (%)", color='red')

    plt.title("Interest Rates vs. House Sales Volume")
    plt.tight_layout()
    plt.show()


# 4. CORRELATION HEATMAP
def plot_correlation_heatmap(df: pd.DataFrame):
    print("\n📊 Plotting simplified correlation heatmap...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr,
        mask=mask,
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# 5. TOP CORRELATIONS
def get_top_correlations(df: pd.DataFrame, n=10):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    top_pairs = (
        upper.stack()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={0: 'correlation'})
    )
    print(f"\n🔗 Top {n} correlated feature pairs:")
    print(top_pairs.head(n))
    return top_pairs.head(n)


# 6. DROP HIGHLY CORRELATED FEATURES
def drop_highly_correlated(df: pd.DataFrame, threshold=0.95, exclude=[]):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold) and col not in exclude
    ]

    print(f"\n📉 Dropping highly correlated features (> {threshold}): {to_drop}")
    return df.drop(columns=to_drop)


# 7. PLOT DISTRIBUTIONS
def plot_numeric_distributions(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n📊 Plotting distributions for numeric columns: {numeric_cols}")
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    plt.figure(figsize=(16, 4 * n_rows))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], bins=30, kde=True, color='skyblue')
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()


# 8. IDENTIFY SKEWED FEATURES
def identify_skewed_features(df: pd.DataFrame, threshold=0.8):
    numeric_df = df.select_dtypes(include=[np.number])
    skewness = numeric_df.skew().sort_values(ascending=False)
    skewed = skewness[skewness > threshold]
    print(f"\n📈 Skewed features (>|{threshold}|):\n{skewed}")
    return skewed.index.tolist()


# 9. LOG TRANSFORM SKEWED FEATURES
def log_transform_selected(df: pd.DataFrame, exclude_keywords=['change'], threshold=0.8):
    df = df.copy()
    numeric_df = df.select_dtypes(include=[np.number])
    skewed = numeric_df.skew().sort_values(ascending=False)
    transformed_cols = []

    for col, skew in skewed.items():
        if any(kw in col.lower() for kw in exclude_keywords):
            continue
        if skew > threshold and (df[col] > 0).all():
            new_col = f"{col}_log"
            df[new_col] = np.log1p(df[col])
            transformed_cols.append(new_col)
            print(f"✅ Log-transformed: {col} → {new_col}")

    if not transformed_cols:
        print("⚠️ No skewed features transformed.")
    return df


# 10. COMBINED DISTRIBUTION & BOXPLOT
def plot_distribution_and_boxplot(df: pd.DataFrame, features: list):
    n_cols = 2
    n_rows = len(features)
    plt.figure(figsize=(12, 4 * n_rows))

    for i, col in enumerate(features):
        plt.subplot(n_rows, n_cols, 2 * i + 1)
        sns.histplot(df[col], bins=30, kde=True, color='skyblue')
        plt.title(f"Histogram of {col}")

        plt.subplot(n_rows, n_cols, 2 * i + 2)
        sns.boxplot(x=df[col], color='salmon')
        plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()
