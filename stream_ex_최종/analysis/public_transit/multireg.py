# analysis/population_car/multireg.py

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline


def run_multireg(df):
    X = df[['bus', 'subway', 'taxi']]
    y = df['car_diff_year']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    # ------------------
    # 1️⃣ 단순 다항 회귀 (과적합 확인용)
    # ------------------
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly  = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    results.append({
        "model": "Poly(deg=2) Linear",
        "train_r2": model.score(X_train_poly, y_train),
        "test_r2": model.score(X_test_poly, y_test)
    })

    # ------------------
    # 2️⃣ Ridge + 다항 + 표준화
    # ------------------
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ridge_results = []

    for a in alphas:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=a))
        ])
        pipe.fit(X_train, y_train)

        ridge_results.append({
            "model": "Ridge(deg=2)",
            "alpha": a,
            "train_r2": pipe.score(X_train, y_train),
            "test_r2": pipe.score(X_test, y_test)
        })

    # test R2 기준 best alpha
    best = max(ridge_results, key=lambda x: x["test_r2"])
    best_alpha = best["alpha"]

    # ------------------
    # 3️⃣ 차수 비교 (1차 vs 2차)
    # ------------------
    degree_results = []

    for d in [1, 2]:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha))
        ])
        pipe.fit(X_train, y_train)

        degree_results.append({
            "model": f"Ridge(deg={d})",
            "alpha": best_alpha,
            "train_r2": pipe.score(X_train, y_train),
            "test_r2": pipe.score(X_test, y_test)
        })

    return (
        pd.DataFrame(results),
        pd.DataFrame(ridge_results),
        pd.DataFrame(degree_results),
        best_alpha
    )
