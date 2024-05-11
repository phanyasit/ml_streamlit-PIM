import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import numpy as np
import pandas as pd

st.title('Iris Data Classification!')
df = px.data.iris()

itrain = np.r_[0:25,50:75,100:125] #แบ่ง train แบบ fix ทุกๆ 25 ดอก เพราะรู้ข้อมูลว่ามี 150 ดอก
itest = np.r_[25:50,75:100,125:150]

xtrain = df.iloc[itrain,:4]
ytrain = df.iloc[itrain, 4]
xtest = df.iloc[itest, :4]
ytest = df.iloc[itest, 4]
K = list(range(1, len(xtrain)+1))
k = st.select_slider('Select k',
                     options=list(range(1, len(xtrain)+1)))
cls = KNeighborsClassifier(n_neighbors=k)

cls.fit(xtrain,ytrain)

ztest = cls.predict(xtest)


acc = np.sum(ytest==ztest)/len(ytest)
st.write(f'Accuracy = {acc*100:.2f}%')
# df
ACC = []

for k in K:
    cls = KNeighborsClassifier(n_neighbors=k)
    cls.fit(xtrain,ytrain)
    ztest = cls.predict(xtest)
    ACC.append(np.sum(ytest==ztest)/len(ytest))

df = pd.DataFrame(columns=['k','acc'])
df['k'] = K
df['acc'] = ACC

fig = px.line(df,x=K,y=ACC)
st.plotly_chart(fig)