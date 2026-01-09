# analysis/population_car/regression.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def run_regression(df, selected_district):
    
    df = df[df['district']==selected_district]
    X = df[["population_diff"]]
    y = df["car_diff"]
    # ------------------
    # 기초 통계
    # ------------------
    desc = df[["population_diff", "car_diff"]].describe()
    corr = df[["population_diff", "car_diff"]].corr()

    # ------------------
    # 선형 회귀
    # ------------------
    model_raw = LinearRegression()
    model_raw.fit(X, y)

    coef = model_raw.coef_[0]
    intercept = model_raw.intercept_

    coef_df = pd.DataFrame({
        "Coefficient": [coef],
        "Intercept": [intercept]
    }, index=["population_diff"])

    # ------------------
    # 모델 성능
    # ------------------
    r2 = model_raw.score(X,y)

    # ------------------
    # 시각화
    # ------------------
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.scatter(X, y, alpha=0.4)
    ax.plot(X, model_raw.predict(X), color="red")

    ax.set_xlabel("population_diff")
    ax.set_ylabel("car_diff")

    return fig, desc, corr, coef_df, r2
