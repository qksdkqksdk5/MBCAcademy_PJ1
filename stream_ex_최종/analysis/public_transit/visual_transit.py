import matplotlib.pyplot as plt
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


def run_visual_transit(df):
    set_korean_font()

    # 1️⃣ 버스 vs 자동차 증감
    fig1, ax1 = plt.subplots(figsize=(5.5, 4))
    ax1.scatter(
        df['bus'],
        df['car_diff_year'],
        alpha=0.6,
        edgecolor="k"
    )
    ax1.set_title("버스 이용량 vs 연간 자동차 증감")
    ax1.set_xlabel("버스 이용량")
    ax1.set_ylabel("자동차 증감 수")
    ax1.grid(alpha=0.3)

    # 2️⃣ 버스 vs 지하철
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    ax2.scatter(
        df['bus'],
        df['subway'],
        alpha=0.6,
        edgecolor="k"
    )
    ax2.set_title("버스 이용량 vs 지하철 이용량")
    ax2.set_xlabel("버스 이용량")
    ax2.set_ylabel("지하철 이용량")
    ax2.grid(alpha=0.3)

    return fig1, fig2






