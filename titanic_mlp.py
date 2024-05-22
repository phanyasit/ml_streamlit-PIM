import seaborn as sns
import streamlit as st
import numpy as np
import torch
from torch import nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 3)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

model = Classifier()
model.load_state_dict(torch.load('model.pth'))
model.eval()

st.title('Iris Dataset')
st.sidebar.title('Input Parameters')
x = np.array([0.] * 4)
for i, col in enumerate(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']):
    x[i] = st.sidebar.slider(col, 0, 8, 0)
st.write(x)
labels = ['setosa', 'versicolor', 'virginica']
with torch.no_grad():
    y_ = model(torch.tensor([x], dtype=torch.float32))
st.write('Predicted class:', labels[y_.argmax(dim=1)[0]])
