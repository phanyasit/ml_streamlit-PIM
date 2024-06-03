import streamlit as st
import pandas as pd
import plotly.express as px

st.title('Exploratory Data Analysis')

if 'df' in st.session_state:
    df = st.session_state['df']

    selected_col = st.sidebar.multiselect('Select columns', df.columns)
    if len(selected_col) == 0:
        selected_col = df.columns
    st.write(df[selected_col])

    corr = df[selected_col].corr().round(2)
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig)

    col = st.sidebar.selectbox('Select a column', df.columns)
    tmp = df[col]
    if pd.api.types.is_numeric_dtype(tmp):
        outliers = st.sidebar.checkbox('Outliers', False)
        if outliers:
            q_low = tmp.quantile(0.01)
            q_high = tmp.quantile(0.99)
            tmp = tmp[(tmp > q_low) & (tmp < q_high)]
        st.write(tmp.describe())
        fig = px.histogram(tmp, x=col)
        st.plotly_chart(fig)
    else:
        st.write(tmp.value_counts())
        fig = px.pie(tmp, names=col)
        st.plotly_chart(fig)