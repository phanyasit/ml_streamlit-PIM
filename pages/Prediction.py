import streamlit as st
import pandas as pd
import joblib

# เรียก model ใน session state ใน ram ไม่ใช่ใน disk
# st.title('Prediction')
# # model = joblib.load('model.joblib')
# if 'model' in st.session_state:
#     model = st.session_state['model']
#     st.write(model)

st.title('Prediction')
model = joblib.load('model.joblib')
st.write(model)

if 'df' in st.session_state:
    df = st.session_state['df']
    target = 'rent amount (R$)'
    cols = df.drop(target, axis=1).columns
    x = pd.DataFrame(columns=cols)
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            x.loc[0, col] = st.sidebar.number_input(col, value=round(df[col].mean()), step=1)
        else:
            x.loc[0, col] = st.sidebar.selectbox(col, df[col].unique())
    st.write(x)
    y = model.predict(x)
    st.write(f'Predicted rent amount: {y[0]:.2f}')



