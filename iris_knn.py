import seaborn as sns
import streamlit as st
import numpy as np
st.title('Iris Dataset')
df = sns.load_dataset('iris')
st.sidebar.title('Input Parameters')
x = np.array([0.] * 4)
for i, col in enumerate(df.columns):
    if col != 'species':
        x[i] = st.sidebar.slider(col, .9*df[col].min(), 1.1*df[col].max(), df[col].mean())
st.write(x)
Xtrain = df.iloc[:, :-1].values
Ytrain = df.iloc[:, -1].values
d = np.sum((Xtrain - x)**2, axis=1)
st.write('Predicted class:', Ytrain[d.argmin()])
