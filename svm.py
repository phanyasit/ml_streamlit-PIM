import streamlit as st
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

tmp = load_iris(as_frame=True)
X = tmp['data']
Y = tmp['target']
itrain = np.r_[0:25, 50:75, 100:125]
itest = np.r_[25:50, 75:100, 125:150]
xtrain = X.iloc[itrain,:]
ytrain = Y.iloc[itrain]
xtest = X.iloc[itest,:]
ytest = Y.iloc[itest]

kernel = st.selectbox('Select kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
C = st.select_slider('Select C',
                     options=[0.1,1,10,100,1000,10000])

cls = SVC(kernel=kernel, C=C)
cls.fit(xtrain,ytrain)
ztest = cls.predict(xtest)
acc = np.sum(ytest == ztest)/len(ytest)
st.write(f'Accuracy = {acc*100:.2f}%')