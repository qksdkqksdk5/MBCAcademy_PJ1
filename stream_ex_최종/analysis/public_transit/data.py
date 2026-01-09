# analysis/population_car/data.py

import pandas as pd
from sqlalchemy import create_engine
import streamlit as st

@st.cache_data
def load_data_transit():
    engine = create_engine(
        "mysql+pymysql://root:12341234@localhost:3306/miniproject"
    )

    df = pd.read_sql(
        "select * from public_transit where district not in ('전체');",
        engine
    )
    df = df[['bus', 'subway', 'taxi', 'car_diff_year']]

    return df
