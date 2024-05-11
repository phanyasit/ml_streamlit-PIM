import seaborn as sns
import streamlit as st

st.title('Datasets')
ds = st.selectbox('Select Dataset',
                  sns.get_dataset_names())
df = sns.load_dataset(ds)

df
