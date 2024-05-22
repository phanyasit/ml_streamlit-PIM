import seaborn as sns
import streamlit as st
import numpy as np
import torch
from torch import nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

model = Classifier()
model.load_state_dict(torch.load('model_titanic'))
model.eval()

st.title('Titanic Dataset')
st.sidebar.title('Input Parameters')
x = np.array([0.] * 4)
for i, col in enumerate(['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male',
       'embarked_C', 'embarked_Q', 'embarked_S', 'who_child', 'who_man',
       'who_woman']):
    x[i] = st.sidebar.slider(col, 0, 8, 0)
st.write(x)
labels = ['รอด', 'ไม่รอด']
with torch.no_grad():
    y_ = model(torch.tensor([x], dtype=torch.float32))
st.write('Predicted class:', labels[y_.argmax(dim=1)[0]])
