"""Data pipeline for the Retail Inventory Restocking IDSS.

Loads the UCI Online Retail dataset, cleans it, aggregates to weekly demand
per StockCode, derives product categories from Description text, and builds
the model feature table.

Run directly to produce data/processed/weekly_features.csv:

    python -m src.data_pipeline
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Online Retail.xlsx")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "weekly_features.csv")
CATEGORY_MAP_PATH = os.path.join(PROCESSED_DIR, "category_map.csv")

N_CATEGORIES = 18

# Major UK holidays covering the dataset span (Dec 2010 - Dec 2011).
UK_HOLIDAYS = [
    "2010-12-25", "2010-12-26", "2010-12-27", "2010-12-28",  # Christmas/Boxing Day (+substitutes)
    "2011-01-01", "2011-01-03",                              # New Year (+substitute)
    "2011-04-22", "2011-04-25",                              # Good Friday, Easter Monday
    "2011-04-29",                                            # Royal Wedding bank holiday
    "2011-05-02", "2011-05-30",                              # Early May / Spring bank holidays
    "2011-08-29",                                            # Summer bank holiday
    "2011-12-25", "2011-12-26", "2011-12-27",                # Christmas/Boxing Day (+substitute)
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw UCI Online Retail Excel file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download the UCI Online Retail "
            "dataset (https://archive.ics.uci.edu/dataset/352/online+retail) "
            "and place 'Online Retail.xlsx' in data/raw/."
        )
    return pd.read_excel(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules from the project spec.

    Drops cancelled invoices (InvoiceNo starting with 'C'), rows with missing
    StockCode/Description, and rows with non-positive UnitPrice (data errors).
    Negative quantities (returns) are dropped from the demand target but
    preserved in a `weekly_returns` count via the `is_return` flag.
    """
    df = df.dropna(subset=["StockCode", "Description"]).copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    # Flag returns before dropping cancelled invoices: in this dataset returns
    # are recorded as negative quantities on "C" invoices, so the flag must be
    # computed first or the returns count comes out empty.
    df["is_return"] = df["Quantity"] < 0
    df = df[df["is_return"] | ~df["InvoiceNo"].str.startswith("C")]
    df = df[df["UnitPrice"] > 0]
    return df


def _week_start(dates: pd.Series) -> pd.Series:
    """Map timestamps to the Monday of their ISO calendar week."""
    return (dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")).dt.normalize()


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactions to one row per (StockCode, ISO week).

    Demand is the sum of positive quantities; returns are counted separately.
    Weeks with no sales between a product's first sale and the end of the
    dataset are filled in with zero demand so rolling features see true gaps.
    """
    df = df.copy()
    df["week_start"] = _week_start(df["InvoiceDate"])

    sales = df[~df["is_return"]]
    returns = df[df["is_return"]]

    weekly = sales.groupby(["StockCode", "week_start"]).agg(
        units_sold=("Quantity", "sum"),
        unit_price=("UnitPrice", "median"),
        description=("Description", "first"),
    ).reset_index()

    ret = returns.groupby(["StockCode", "week_start"])["Quantity"].sum().abs()
    ret.name = "weekly_returns"
    weekly = weekly.merge(ret, on=["StockCode", "week_start"], how="left")
    weekly["weekly_returns"] = weekly["weekly_returns"].fillna(0)

    # Fill zero-demand weeks from each product's first sale to the dataset end.
    all_weeks = pd.date_range(weekly["week_start"].min(), weekly["week_start"].max(), freq="W-MON")
    first_week = weekly.groupby("StockCode")["week_start"].min()
    full_index = pd.concat(
        [
            pd.DataFrame({"StockCode": code, "week_start": all_weeks[all_weeks >= start]})
            for code, start in first_week.items()
        ],
        ignore_index=True,
    )
    weekly = full_index.merge(weekly, on=["StockCode", "week_start"], how="left")
    weekly["units_sold"] = weekly["units_sold"].fillna(0)
    weekly["weekly_returns"] = weekly["weekly_returns"].fillna(0)
    # Carry price/description through zero-sale weeks.
    weekly = weekly.sort_values(["StockCode", "week_start"])
    weekly[["unit_price", "description"]] = (
        weekly.groupby("StockCode")[["unit_price", "description"]].ffill().bfill()
    )
    return weekly.reset_index(drop=True)


def assign_categories(descriptions: pd.Series, n_categories: int = N_CATEGORIES,
                      cache_path: str = CATEGORY_MAP_PATH) -> pd.DataFrame:
    """Assign each StockCode a product category via TF-IDF + KMeans on its
    Description. The mapping is cached to CSV so it is computed once.

    `descriptions` is indexed by StockCode with one representative description
    per product. Returns a DataFrame with StockCode, category, category_name.
    """
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, dtype={"StockCode": str})
        if set(descriptions.index) <= set(cached["StockCode"]):
            return cached

    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = descriptions.astype(str).str.upper().str.replace(r"[^A-Z ]", " ", regex=True)
    vec = TfidfVectorizer(stop_words="english", min_df=5)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=n_categories, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Name each cluster after its two highest-weight TF-IDF terms.
    terms = np.array(vec.get_feature_names_out())
    names = {}
    for k in range(n_categories):
        top = terms[np.argsort(km.cluster_centers_[k])[::-1][:2]]
        names[k] = "_".join(t.upper() for t in top)

    mapping = pd.DataFrame({
        "StockCode": descriptions.index,
        "category": labels,
        "category_name": [names[l] for l in labels],
    })
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    mapping.to_csv(cache_path, index=False)
    return mapping


def _holiday_weeks() -> set:
    dates = pd.to_datetime(pd.Series(UK_HOLIDAYS))
    return set(_week_start(dates))


def build_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature table, one row per (StockCode, week).

    Rolling features are shifted by one week so they never include the target
    week's own sales. Products with fewer than 4 weeks of prior history fall
    back to their category-level averages (cold start handling).
    """
    df = weekly.sort_values(["StockCode", "week_start"]).copy()

    iso = df["week_start"].dt.isocalendar()
    df["week_of_year"] = iso.week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["is_major_holiday_week"] = df["week_start"].isin(_holiday_weeks()).astype(int)

    g = df.groupby("StockCode")["units_sold"]
    df["avg_weekly_sales_past_4_weeks"] = g.transform(
        lambda s: s.shift(1).rolling(4, min_periods=4).mean()
    )
    roll_mean_12 = g.transform(lambda s: s.shift(1).rolling(12, min_periods=4).mean())
    roll_std_12 = g.transform(lambda s: s.shift(1).rolling(12, min_periods=4).std())
    df["coefficient_of_variation_weekly_demand_past_12_weeks"] = np.where(
        roll_mean_12 > 0, roll_std_12 / roll_mean_12, 0.0
    )
    df["avg_weekly_sales_same_week_last_year"] = g.shift(52)
    df["weeks_of_history"] = df.groupby("StockCode").cumcount()

    # Cold start fallback: category-level averages for rolling features where
    # a product has fewer than 4 weeks of prior history.
    cat = df.groupby(["category", "week_start"], as_index=False).agg(cat_units=("units_sold", "mean"))
    cat = cat.sort_values(["category", "week_start"])
    cg = cat.groupby("category")["cat_units"]
    cat["cat_avg_4w"] = cg.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    cat_mean_12 = cg.transform(lambda s: s.shift(1).rolling(12, min_periods=2).mean())
    cat_std_12 = cg.transform(lambda s: s.shift(1).rolling(12, min_periods=2).std())
    cat["cat_cv_12w"] = np.where(cat_mean_12 > 0, cat_std_12 / cat_mean_12, 0.0)
    cat["cat_avg_lastyear"] = cg.shift(52)
    df = df.merge(cat.drop(columns=["cat_units"]), on=["category", "week_start"], how="left")

    cold = df["weeks_of_history"] < 4
    for col, cat_col in [
        ("avg_weekly_sales_past_4_weeks", "cat_avg_4w"),
        ("coefficient_of_variation_weekly_demand_past_12_weeks", "cat_cv_12w"),
    ]:
        df.loc[cold | df[col].isna(), col] = df.loc[cold | df[col].isna(), cat_col]

    # The dataset spans barely more than one year, so the same-week-last-year
    # lag is missing for most rows; fall back to the category average for that
    # week last year, then to the product's own 4-week average.
    df["avg_weekly_sales_same_week_last_year"] = (
        df["avg_weekly_sales_same_week_last_year"]
        .fillna(df["cat_avg_lastyear"])
        .fillna(df["avg_weekly_sales_past_4_weeks"])
    )

    for col in ["avg_weekly_sales_past_4_weeks",
                "coefficient_of_variation_weekly_demand_past_12_weeks",
                "avg_weekly_sales_same_week_last_year"]:
        df[col] = df[col].fillna(0.0)

    df = df.drop(columns=["cat_avg_4w", "cat_cv_12w", "cat_avg_lastyear"])
    return df


def run_pipeline(raw_path: str = RAW_PATH, out_path: str = FEATURES_PATH) -> pd.DataFrame:
    """Full pipeline: load -> clean -> weekly aggregate -> categorize -> features."""
    raw = load_raw(raw_path)
    cleaned = clean(raw)
    weekly = aggregate_weekly(cleaned)

    desc = weekly.groupby("StockCode")["description"].first()
    categories = assign_categories(desc)
    weekly = weekly.merge(categories, on="StockCode", how="left")
    weekly["category"] = weekly["category"].fillna(-1).astype(int)

    features = build_features(weekly)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    features.to_csv(out_path, index=False)
    return features


if __name__ == "__main__":
    feats = run_pipeline()
    print(f"Saved feature table: {FEATURES_PATH}")
    print(f"Rows: {len(feats):,} | Products: {feats['StockCode'].nunique():,} | "
          f"Weeks: {feats['week_start'].nunique()}")
    print(f"Date range: {feats['week_start'].min().date()} to {feats['week_start'].max().date()}")
    print("\nColumn summary:")
    print(feats.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
    print("\nSample rows:")
    cols = ["StockCode", "week_start", "units_sold", "category_name",
            "avg_weekly_sales_past_4_weeks", "coefficient_of_variation_weekly_demand_past_12_weeks",
            "avg_weekly_sales_same_week_last_year", "unit_price", "is_major_holiday_week"]
    print(feats[feats["weeks_of_history"] >= 8][cols].head(8).to_string(index=False))
