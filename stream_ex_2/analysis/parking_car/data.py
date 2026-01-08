# analysis/population_car/data.py

import pandas as pd
from sqlalchemy import create_engine
import streamlit as st

@st.cache_data
def load_data_parking():
    engine = create_engine(
        "mysql+pymysql://root:12341234@localhost:3306/miniproject2"
    )

    df = pd.read_sql(
        "select * from parking_car;",
        engine
    )

    return df
