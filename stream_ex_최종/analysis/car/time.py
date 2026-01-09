import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

# ------------------
# 한글 폰트 설정
# ------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def plot_monthly(df):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['car_count_month'], marker='o', linewidth=2)
    ax.set_title("서울시 월별 자동차 등록 현황")
    ax.set_xlabel("날짜")
    ax.set_ylabel("등록 차량 수")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    return fig


def plot_diff_1(df):
    diff_1 = df['car_count_month'].diff().dropna()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(diff_1, marker='o', linewidth=2)
    ax.axhline(0, linestyle='--', alpha=0.7)
    ax.set_title("서울시 월별 자동차 등록 수 1차 차분")
    ax.set_xlabel("날짜")
    ax.set_ylabel("전월 대비 증감")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    return fig, diff_1


def stationarity_test(diff_1):
    adf = adfuller(diff_1)
    kpss_result = kpss(diff_1, regression='c', nlags='auto')

    return {
        "adf_stat": adf[0],
        "adf_p": adf[1],
        "adf_crit": adf[4],
        "kpss_stat": kpss_result[0],
        "kpss_p": kpss_result[1],
        "kpss_crit": kpss_result[3],
    }


def fit_arima(df, order=(1, 1, 1)):
    model = ARIMA(
        df['car_count_month'],
        order=order
    )
    result = model.fit()
    return result

def forecast_12_months(result, last_date):

    last_date = pd.to_datetime(last_date)
    forecast = result.get_forecast(steps=12)
    mean = forecast.predicted_mean
    conf = forecast.conf_int()

    index = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=12,
        freq='MS'
    )

    mean.index = index
    conf.index = index

    return mean, conf

def plot_forecast(df, forecast_mean, conf_int):
    fig, ax = plt.subplots(figsize=(14, 6))

    # 실제 값
    ax.plot(
        df.index,
        df['car_count_month'],
        label="Observed",
        linewidth=2
    )

    # 예측 값
    ax.plot(
        forecast_mean.index,
        forecast_mean,
        linestyle="--",
        label="Forecast (12 months)"
    )

    # 신뢰구간
    ax.fill_between(
        forecast_mean.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.2,
        label="95% Confidence Interval"
    )

    ax.set_title("서울시 자동차 등록 대수 12개월 예측")
    ax.set_xlabel("날짜")
    ax.set_ylabel("등록 차량 수")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig