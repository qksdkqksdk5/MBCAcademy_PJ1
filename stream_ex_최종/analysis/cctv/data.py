# analysis/population_car/data.py

import pandas as pd
import streamlit as st

@st.cache_data
def load_data_cctv():
    df = pd.read_csv('data/cctv_accident.csv')

    return df
