# analysis/population_car/data.py

import pandas as pd
from sqlalchemy import create_engine
import streamlit as st

@st.cache_data
def load_data_traffic():
    engine = create_engine(
        "mysql+pymysql://root:12341234@localhost:3306/miniproject2"
    )

    df = pd.read_sql(
        "select * from vehicle;",
        engine
    )

    df_traffic = pd.read_sql(
        "select * from traffic;",
        engine
    )

    return df, df_traffic
