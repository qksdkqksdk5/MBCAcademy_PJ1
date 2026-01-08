# analysis/population_car/ridge.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


def run_ridge(df):

     # ------------------
    # 변수 설정
    # ------------------
    X = df[['car_count']].values
    y = df['parking_area'].values

    # ------------------
    # Train / Test split
    # ------------------
    train_input, test_input, train_target, test_target = train_test_split(
        X, y, random_state=42
    )
    # ------------------
    # 표준화
    # ------------------
    scaler = StandardScaler()
    scaler.fit(train_input)

    train_scaled = scaler.transform(train_input)
    test_scaled = scaler.transform(test_input)

    # ------------------
    # alpha별 성능 확인
    # ------------------
    alpha_list = [0.001, 0.1, 1, 100, 10000, 1_000_000]
    train_score = []
    test_score = []

    for alpha in alpha_list:
        ridge = Ridge(alpha=alpha)
        ridge.fit(train_scaled, train_target)

        train_score.append(ridge.score(train_scaled, train_target))
        test_score.append(ridge.score(test_scaled, test_target))

    # ------------------
    # 시각화
    # ------------------
    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(np.log10(alpha_list), train_score, label="Train R²")
    ax.plot(np.log10(alpha_list), test_score, label="Test R²")

    ax.set_xlabel("log10(alpha)")
    ax.set_ylabel("R² Score")
    ax.set_title("Ridge 규제 강도에 따른 성능 변화")
    ax.legend()
    ax.grid(True)

    # ------------------
    # 최적 alpha 선택 (test R² 기준)
    # ------------------
    best_idx = np.argmax(test_score)
    best_alpha = alpha_list[best_idx]

    best_ridge = Ridge(alpha=best_alpha)
    best_ridge.fit(train_scaled, train_target)

    best_scores = {
        "best_alpha": best_alpha,
        "train_r2": best_ridge.score(train_scaled, train_target),
        "test_r2": best_ridge.score(test_scaled, test_target)
    }

    return fig, best_scores
