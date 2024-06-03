import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, OneHotEncoder
from sklearn.feature_selection import f_regression, r_regression, mutual_info_regression, SelectKBest, SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
import joblib

st.title('Regression')
if 'df' in st.session_state:
    target = 'rent amount (R$)'
    df = st.session_state['df']
    drop = st.sidebar.selectbox('Dummy variable trap', ['None', 'first', 'if_binary'])
    drop = None if drop == 'None' else drop

    scaler_x = st.sidebar.selectbox('Feature scaling',
                                    ['None', 'StandardScaler', 'MinMaxScaler', 'QuantileTransformer'])
    scaler_y = st.sidebar.selectbox('Target scaling',
                                    ['None', 'StandardScaler', 'MinMaxScaler', 'QuantileTransformer'])
    scaler_x = 'passthrough' if scaler_x == 'None' else eval(scaler_x + '()')
    scaler_y = None if scaler_y == 'None' else eval(scaler_y + '()')

    numeric_features = df.drop(target, axis=1).select_dtypes(include='number').columns
    categorical_features = df.drop(target, axis=1).select_dtypes(exclude='number').columns

    numeric_transformer = Pipeline(steps=[
        ('scaler', scaler_x)
    ])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop=drop, handle_unknown='ignore'))])

    selector = st.sidebar.selectbox('Feature selection',
                                    ['None', 'f_regression', 'r_regression', 'mutual_info_regression',
                                     'SelectFromModel'])
    if selector in ['f_regression', 'r_regression', 'mutual_info_regression']:
        k = st.sidebar.slider('Number of features', 1, len(df.columns), 1)
    if selector == 'None':
        selector = 'passthrough'
    else:
        if selector == 'SelectFromModel':
            selector = SelectFromModel(RandomForestRegressor())
        else:
            selector = SelectKBest(eval(selector), k=k)

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    regressor = st.sidebar.selectbox('Regressor', ['LinearRegression', 'RandomForestRegressor', 'SVR'])
    regressor = eval(regressor + '()')

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('regressor', regressor)
    ])

    target_transformer = TransformedTargetRegressor(regressor=pipe, transformer=scaler_y)

    with st.spinner('Training model...'):
        target_transformer.fit(df.drop(target, axis=1), df[target])
    Z = target_transformer.predict(df.drop(target, axis=1))
    st.subheader(f'MAE: {mean_absolute_error(df[target], Z): .05f}')

    tmp = pd.DataFrame({'Actual': df[target], 'Predicted': Z})
    fig = px.line(tmp, y=['Actual', 'Predicted'], title='Actual vs Predicted')
    st.plotly_chart(fig)

    st.sidebar.divider()
    if st.sidebar.button('Save model'):
        joblib.dump(target_transformer, 'model.joblib')

    # Save เก็บ model ไว้ใน session state ใน ram ไม่ใช่ใน disk
    # st.sidebar.divider()
    # if st.sidebar.button('Save model'):
    #     # joblib.dump(target_transformer, 'model.joblib')
    #     st.session_state['model'] = target_transformer