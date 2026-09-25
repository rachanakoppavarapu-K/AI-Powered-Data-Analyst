"""One-off script to generate the three reference datasets used for testing."""
import pathlib
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
out = pathlib.Path(__file__).parent

# ── 1. Classification: Titanic-style passenger survival ─────────────────────
n = 300
titanic_like = pd.DataFrame(
    {
        "PassengerId": range(1, n + 1),
        "Survived": rng.integers(0, 2, n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Age": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(1, 80, n).round(1)),
        "Fare": rng.uniform(5, 500, n).round(2),
        "Sex": rng.choice(["male", "female"], n),
        "Embarked": rng.choice(["C", "Q", "S", None], n),
    }
)
titanic_like.to_csv(out / "titanic_like.csv", index=False)

# ── 2. Regression: House price prediction ───────────────────────────────────
n = 300
house = pd.DataFrame(
    {
        "Id": range(1, n + 1),
        "LotArea": rng.integers(3000, 20000, n),
        "YearBuilt": rng.integers(1900, 2023, n),
        "OverallQual": rng.integers(1, 11, n),
        "GrLivArea": rng.integers(600, 4000, n),
        "BedroomAbvGr": rng.integers(1, 6, n),
        "Neighborhood": rng.choice(["CollgCr", "Veenker", "Crawfor", "NoRidge"], n),
        "SalePrice": (
            rng.integers(60000, 400000, n)
        ),
    }
)
house.to_csv(out / "house_prices.csv", index=False)

# ── 3. Clustering: Customer segmentation (no obvious target) ─────────────────
n = 300
customers = pd.DataFrame(
    {
        "CustomerID": range(1, n + 1),
        "Annual Income (k$)": rng.integers(15, 140, n),
        "Spending Score (1-100)": rng.integers(1, 101, n),
        "Age": rng.integers(18, 70, n),
        "Gender": rng.choice(["Male", "Female"], n),
    }
)
customers.to_csv(out / "customers.csv", index=False)

print("Datasets written to", out)
