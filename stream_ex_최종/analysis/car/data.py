# analysis/population_car/data.py

import pandas as pd
from sqlalchemy import create_engine
import streamlit as st

@st.cache_data
def load_data_car_month():
    engine = create_engine(
        "mysql+pymysql://root:12341234@localhost:3306/miniproject"
    )

    df = pd.read_sql(
        "select * from car_month;",
        engine
    )
    # datetime → datetime 타입으로 변환
    df['datetime'] = pd.to_datetime(df['datetime'])

    # datetime을 index로 설정
    df = df.set_index('datetime').sort_index()

    # 주기 명시
    df = df.asfreq('MS')
    return df
