import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import matplotlib.font_manager as fm
import numpy as np

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
# ------------------
# 한글 폰트 설정
# ------------------
def set_korean_font():
    try:
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc("font", family=font_name)
        plt.rcParams["axes.unicode_minus"] = False
    except:
        pass


# ------------------
# 1. 상관분석 + 산점도
# ------------------
def plot_correlation(df):
    x = df["car_count"].values
    y = df["parking_area"].values

    r, p = stats.pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(x, y, alpha=0.6)
    ax.set_xlabel("car_count")
    ax.set_ylabel("parking_area")
    ax.set_title(f"Correlation (r = {r:.3f})")

    return fig, r, p


# ------------------
# 2. 단순 선형 회귀 + 성능
# ------------------
def run_parking_regression(df):
    X = df[["car_count"]].values
    y = df["parking_area"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 성능 지표
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "mae": mean_absolute_error(y_test, y_test_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred))
    }

    # 시각화
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(X_train, y_train, alpha=0.5)

    x_min, x_max = X_train.min(), X_train.max()
    ax.plot(
        [x_min, x_max],
        [x_min * model.coef_[0] + model.intercept_,
         x_max * model.coef_[0] + model.intercept_],
        color="red",
        linestyle="--"
    )

    ax.set_xlabel("car_count")
    ax.set_ylabel("parking_area")
    ax.set_title("Linear Regression")

    return fig, model, metrics


# ------------------
# 3. 시간 기반 예측
# ------------------
def predict_future(df):
    start_year = 2011
    df['year'] = range(start_year, start_year + len(df))

    X = df[['year']]
    y_car = df['car_count']
    y_parking = df['parking_area']

    model_car = LinearRegression().fit(X, y_car)
    model_parking = LinearRegression().fit(X, y_parking)

    future = pd.DataFrame({'year': [2025]})

    pred_car = int(model_car.predict(future)[0])
    pred_parking = int(model_parking.predict(future)[0])

    return {
        "year": 2025,
        "pred_car": pred_car,
        "pred_parking": pred_parking,
        "parking_ratio": pred_parking / pred_car * 100
    }






