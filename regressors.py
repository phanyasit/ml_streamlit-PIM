import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

x = np.random.rand(200)
y = 3 * np.sin(2 * x + 1 ) + np.random.normal( loc:0, scale:0.1, len(x))

fig, ax = plt.subplots()
ax.plot(x,y,'-')
st.pyplot(fig)

k = st.select_slider('select k', options=range(1,11))
rgs = KNeighborsRegressor(n_neighbors=k)
rgs.fit(x[:, None], y)
z = rgs.predict(x[:, None])
ax.plot(x,z, '.r')

st.pyplot(fig)