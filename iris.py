import seaborn as sns
import streamlit as st
import pandas as pd
st.title('Iris Dataset')
# df = sns.load_dataset('iris')
url = 'https://archive.ics.uci.edu/ml/' + \
      'machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df.columns = ['sepal_length', 'sepal_width',
              'petal_length', 'petal_width', 'species']

x = st.selectbox('select X-axis', df.columns[:-1])
y = st.selectbox('select Y-axis', df.columns[:-1])
st.write('You selected:', x, y)
st.scatter_chart(df, x=x, y=y, color=df.columns[-1])
