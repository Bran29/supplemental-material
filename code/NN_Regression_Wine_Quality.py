import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("ggplot")  #using style ggplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale

# data preprocess



""" Torch training """

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



class NN_Module(nn.Module):
    def __init__(self, input_dim=8, output_dim=1, device=None):
        super(NN_Module, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)

    def extract(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def train(model, loader, optimizer, loss_fn, epochs=30, device=torch.device('cpu')):
    model.train()
    model = model.to(device)
    for epoch in range(int(epochs)):
        for i, (batch_data, batch_target) in enumerate(loader):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)

            optimizer.zero_grad()
            mse_loss = loss_fn(model(batch_data), batch_target.view(-1, 1))
            mse_loss.backward()
            optimizer.step()
    return model

def evaluate(model, valid_loader, loss_fn, device=None):
    valid_loss = 0
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for valid_i, (data, targets) in enumerate(valid_loader):
            data, targets = data.to(device), targets.to(device)
            pred = model(data)
            valid_loss += loss_fn(model(data), targets.view(-1, 1))

    return valid_loss.item()

def loss_result(X_train,y_train,X_test,y_test):
    tensor_x = torch.Tensor(X_train)  # transform to torch tensor
    tensor_y = torch.Tensor(y_train)

    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    loader = DataLoader(dataset, batch_size=64)  # create your dataloader


    tensor_x = torch.Tensor(X_test)  # transform to torch tensor
    tensor_y = torch.Tensor(y_test)

    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    valid_loader = DataLoader(dataset, batch_size=1000)  # create your dataloader


    device = torch.device('cpu')
    model = NN_Module(input_dim=10, output_dim=1,device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.MSELoss()
    E = 50

    valid_loss_old = evaluate(model, valid_loader, loss_fn, device)
    model = train(model, loader, optimizer, loss_fn, E, device)
    valid_loss = evaluate(model, valid_loader, loss_fn, device)
    return valid_loss,valid_loss_old

if __name__=="__main__":
    df = pd.read_csv("data/wine_quality/WineQT.csv")
    df.drop('Id', axis=1, inplace=True)
    df1 = df[df['pH'] <= 3.2]
    df2 = df[df['pH'] > 3.2]
    X1 = df1.drop(columns=["quality", "pH"])
    y1 = df1["pH"]
    X2 = df2.drop(columns=["quality", "pH"])
    y2 = df2["pH"]

    X1 = X1.to_numpy()
    X2 = X2.to_numpy()
    y1 = y1.to_numpy()
    y2 = y2.to_numpy()

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=20)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=80)

    valid_loss1,valid_loss_old1=loss_result(X_train1, y_train1, X_test1, y_test1)
    valid_loss2, valid_loss_old2 = loss_result(X_train2, y_train2, X_test2, y_test2)
    print(valid_loss1,valid_loss_old1)
    print(valid_loss2, valid_loss_old2)
