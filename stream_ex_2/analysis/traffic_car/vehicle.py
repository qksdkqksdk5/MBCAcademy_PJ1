# traffic.py
import pandas as pd
import matplotlib.pyplot as plt

# ------------------
# 한글 폰트 설정
# ------------------
def set_korean_font():
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False


# ------------------
# 차종별 합계 컬럼 생성
# ------------------
def add_vehicle_totals(df):
    df = df.copy()

    for cat in ['승용', '승합', '화물', '특수']:
        df[f'{cat}합계'] = df.filter(like=cat).sum(axis=1)

    df['등록합계'] = df.filter(like='합계').sum(axis=1)
    return df


# ------------------
# 년월별 합계 행 생성
# ------------------
def make_monthly_summary(df):
    df = add_vehicle_totals(df)  # ⭐ 여기서 다시 한 번 보장

    dfs = {date: df[df['년월'] == date] for date in df['년월'].unique()}
    summary_rows = []

    for date, target_df in dfs.items():
        sum_values = target_df[
            ['승용합계', '승합합계', '화물합계', '특수합계', '등록합계']
        ].sum()

        sum_row = {'년월': date}
        sum_row.update(sum_values.to_dict())

        summary_rows.append(sum_row)

    return pd.DataFrame(summary_rows)


# ------------------
# 📈 시각화 함수 (fig 반환)
# ------------------
def plot_vehicle_trend(total_summary):
    set_korean_font()

    fig, ax1 = plt.subplots(figsize=(14, 8))

    categories = ['승용합계', '승합합계', '화물합계', '특수합계']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, cat in enumerate(categories):
        ax1.plot(
            total_summary['년월'].astype(str),
            total_summary[cat],
            marker='o',
            label=cat,
            color=colors[i],
            linewidth=2
        )

    ax1.set_xlabel('년월')
    ax1.set_ylabel('차종별 등록 대수')
    ax1.legend(loc='upper left')

    # ▶ 보조축 (전체 등록합계)
    ax2 = ax1.twinx()
    ax2.plot(
        total_summary['년월'].astype(str),
        total_summary['등록합계'],
        color='purple',
        linestyle='--',
        linewidth=3,
        marker='s',
        label='전체 등록합계'
    )

    ax2.set_ylabel('전체 등록합계', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.85))

    ax1.set_xticks(range(len(total_summary)))
    ax1.set_xticklabels(total_summary['년월'].astype(str), rotation=45)

    plt.title('주요 차종 및 전체 등록합계 증감 추이')
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()

    return fig
