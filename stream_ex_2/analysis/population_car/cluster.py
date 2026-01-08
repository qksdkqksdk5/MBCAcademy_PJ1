# analysis/population_car/cluster.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.font_manager as fm


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


def run_clustering(df, selected_district, n_clusters=3):
    set_korean_font()
    # ------------------
    # 자치구 단위 집계
    # 여러 년도를 하나의 자치구 특성 벡터로 압축
    # ------------------
    
    # ------------------
    # 전체 행 제거 (군집은 자치구만)
    # ------------------
    df_cluster_base = df[df["district"] != "전체"].copy()

    # ------------------
    # 자치구별 평균 집계
    # ------------------
    df_cluster = (
        df_cluster_base
        .groupby(["district"], as_index=False)
        .agg({
            "population": "mean",
            "car_count": "mean"
        })
    )

    # ------------------
    # 표준화 (스케일 차이 해결)
    # ------------------
    X = df_cluster[["population", "car_count"]]
    X_scaled = StandardScaler().fit_transform(X)

    # ------------------
    # KMeans
    # ------------------
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df_cluster["cluster"] = kmeans.fit_predict(X_scaled)

    # ------------------
    # 군집 요약
    # ------------------
    summary_df = (
        df_cluster
        .groupby("cluster")[["population", "car_count"]]
        .mean()
        .round(1)
        .reset_index()
    )

    # ------------------
    # Figure 1: 군집별 평균 Bar
    # ------------------
    fig_bar, ax_bar = plt.subplots(figsize=(5, 3.5))

    summary_df.set_index("cluster")[["population", "car_count"]].plot(
        kind="bar",
        ax=ax_bar
    )

    ax_bar.set_title("군집별 평균 인구 수 및 자동차 등록 대수")
    ax_bar.set_xlabel("군집")
    ax_bar.set_ylabel("평균 값")
    ax_bar.legend(["인구 수", "자동차 등록 대수"])

    # ------------------
    # Figure 2: 군집 분포 Scatter
    # ------------------
    fig_scatter, ax_scatter = plt.subplots(figsize=(5, 3.5))

    scatter = ax_scatter.scatter(
        df_cluster["population"],
        df_cluster["car_count"],
        c=df_cluster["cluster"],
        cmap="tab10",
        s=60,
        alpha=0.8
    )

    ax_scatter.set_xlabel("평균 인구 수")
    ax_scatter.set_ylabel("평균 자동차 등록 대수")
    ax_scatter.set_title("자치구별 군집 분포")

    # (선택) 자치구 이름 표시
    # for _, row in df_cluster.iterrows():
    #     ax_scatter.text(
    #         row["population"],
    #         row["car_count"],
    #         row["district"],
    #         fontsize=7,
    #         alpha=0.7
    #     )

    return df_cluster, summary_df, fig_bar, fig_scatter
