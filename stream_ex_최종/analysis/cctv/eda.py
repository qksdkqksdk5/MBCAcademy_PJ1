# analysis/accident/eda.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cctv_vs_death(df):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(df['CCTV설치대수'], df['사고당사망률'])
    ax.set_xlabel('CCTV 설치 대수')
    ax.set_ylabel('사고당 사망률')
    ax.set_title('CCTV 설치 대수 vs 사고당 사망률')
    ax.grid(True)
    return fig


def plot_histograms(df, num_cols):
    fig = df[num_cols].hist(bins=20, figsize=(12,8))
    plt.tight_layout()
    return plt.gcf()


def plot_corr_heatmap(df, num_cols):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(
        df[num_cols].corr(),
        annot=True, fmt=".2f", cmap="coolwarm", ax=ax
    )
    ax.set_title("변수 간 상관계수")
    return fig


def plot_severity_box(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(
        x='심각정도',
        y='사고당사망률',
        data=df,
        order=['낮음','보통','심각'],
        ax=ax
    )
    ax.set_title('심각도별 사고당 사망률 분포')
    return fig
