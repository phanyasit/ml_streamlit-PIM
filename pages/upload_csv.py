import streamlit as st
import pandas as pd

csv = st.file_uploader('Upload CSV', type='csv')
if csv is not None:
    df = pd.read_csv(csv)
    st.dataframe(df)
