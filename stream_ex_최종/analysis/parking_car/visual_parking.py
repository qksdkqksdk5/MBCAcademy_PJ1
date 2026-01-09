from matplotlib import ticker
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

    x_scaled = x / 10000
    y_scaled = y / 10000

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        x_scaled,
        y_scaled,
        color="royalblue",
        s=100,
        alpha=0.7,
        edgecolors="white",
        linewidth=1.5
    )

    ax.set_title(
        f"자동차 수와 주차면 수의 상관관계 (r = {r:.3f})",
        fontsize=18,
        fontweight="bold",
        pad=20
    )
    ax.set_xlabel("자동차 수 (단위: 만 대)", fontsize=14, labelpad=10)
    ax.set_ylabel("주차면 수 (단위: 만 면)", fontsize=14, labelpad=10)

    ax.xaxis.set_major_formatter(
        ticker.StrMethodFormatter("{x:,.0f}")
    )
    ax.yaxis.set_major_formatter(
        ticker.StrMethodFormatter("{x:,.0f}")
    )

    ax.tick_params(axis="both", labelsize=12)

    plt.tight_layout()

    return fig, r, p


# ------------------
# 2. 단순 선형 회귀 + 성능
# ------------------
def run_parking_regression(df):

    X = df[["car_count"]].values
    y = df["parking_area"].values


    X_scaled = X / 10000
    y_scaled = y / 10000

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "mae": mean_absolute_error(y_test, y_test_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred))
    }

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        X_train,
        y_train,
        s=200,
        color="#6c85bd",
        alpha=0.9,
        label="실제 데이터"
    )

    x_min, x_max = X_train.min(), X_train.max()
    y_min = model.predict([[x_min]])[0]
    y_max = model.predict([[x_max]])[0]

    ax.plot(
        [x_min, x_max],
        [y_min, y_max],
        color="#c41e3a",
        linestyle="--",
        linewidth=3,
        label="회귀선"
    )

    ax.set_title(
        "자동차 수와 주차면 수의 선형 회귀 관계",
        fontsize=20,
        fontweight="bold",
        pad=20
    )
    ax.set_xlabel("자동차 수 (단위: 만 대)", fontsize=14, labelpad=10)
    ax.set_ylabel("주차면 수 (단위: 만 면)", fontsize=14, labelpad=10)

    ax.tick_params(axis="both", labelsize=12)

    ax.legend(fontsize=12)
    plt.tight_layout()

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






