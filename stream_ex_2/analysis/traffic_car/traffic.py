import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------
# 한글 폰트 설정
# ------------------
def set_korean_font():
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 연도별 등록대수 요약
# =========================
def make_yearly_summary(total_summary):
    total_summary = total_summary.copy()
    total_summary.columns = total_summary.columns.str.strip()

    total_summary['년월'] = pd.to_numeric(
        total_summary['년월'].astype(str).str.strip(),
        errors='coerce'
    )
    total_summary['연도'] = total_summary['년월'] // 100

    vehicle_columns = ['승용합계', '승합합계', '화물합계', '특수합계', '등록합계']

    yearly_df = (
        total_summary
        .groupby('연도')[vehicle_columns]
        .mean()
        .round(0)
        .astype(int)
        .reset_index()
    )

    yearly_df['전년대비_증감률(%)'] = (
        yearly_df['등록합계'].pct_change() * 100
    ).round(2)

    return yearly_df


# =========================
# 2. 교통량 증감률 막대그래프
# =========================
def plot_traffic_growth_bar(df_traffic):
    categories = df_traffic['구분(유형)'].tolist()
    rate_22 = df_traffic['22증감률(%)'].tolist()
    rate_23 = df_traffic['23증감률(%)'].tolist()
    rate_24 = df_traffic['24증감률(%)'].tolist()

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5, 3.5))

    r1 = ax.bar(x - width, rate_22, width, label='22년')
    r2 = ax.bar(x, rate_23, width, label='23년')
    r3 = ax.bar(x + width, rate_24, width, label='24년')

    ax.set_title('지점 유형별 연도별 교통량 증감률', fontsize=15)
    ax.set_ylabel('증감률 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.axhline(0, color='black')

    for rects in [r1, r2, r3]:
        for r in rects:
            h = r.get_height()
            ax.text(
                r.get_x() + r.get_width()/2,
                h,
                f'{h:.1f}%',
                ha='center',
                va='bottom' if h >= 0 else 'top',
                fontsize=9
            )

    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig


# =========================
# 3. 상관관계 분석 + 시각화
# =========================
def analyze_correlation(total_summary, df_traffic):
    total_summary['연도'] = total_summary['년월'] // 100
    yearly_reg = (
        total_summary
        .groupby('연도')['등록합계']
        .mean()
        .reset_index()
    )
    yearly_reg['등록증감률'] = yearly_reg['등록합계'].pct_change() * 100

    target_years = [2022, 2023, 2024]
    traffic_growth = df_traffic[['22증감률(%)','23증감률(%)','24증감률(%)']].iloc[0].values
    reg_growth = yearly_reg[yearly_reg['연도'].isin(target_years)]['등록증감률'].values

    corr = pd.Series(traffic_growth).corr(pd.Series(reg_growth))

    plot_df = pd.DataFrame({
        '연도': target_years,
        '교통량증감률': traffic_growth,
        '등록증감률': reg_growth
    })

    # 추이 그래프
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(plot_df['연도'], plot_df['등록증감률'], marker='o', label='등록대수')
    ax1.plot(plot_df['연도'], plot_df['교통량증감률'], marker='s', linestyle='--', label='교통량')
    ax1.set_title('교통량 vs 등록대수 증감률 추이')
    ax1.legend()
    ax1.grid(True)

    # 산점도
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    sns.regplot(
        x='교통량증감률',
        y='등록증감률',
        data=plot_df,
        ax=ax2
    )
    ax2.set_title(f'상관계수: {corr:.3f}')

    return corr, fig1, fig2
