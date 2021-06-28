import datetime as datetime
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import datetime
from torch import nn

target_label = 'mood'

# df = pd.read_csv('/home/chrei/Dropbox/uni/forschungsmodul1/master_Daily Summaries_mod_clean_2 (test).csv', index_col=0)
df = pd.read_csv('/home/chrei/Dropbox/uni/forschungsmodul1/master_Daily Summaries_mod_clean_2.csv', index_col=0)

# drop food and manual logged attributes
df = df.drop(
    ['Low latitude (deg)', 'Low longitude (deg)', 'High latitude (deg)', 'High longitude (deg)', 'Average weight (kg)',
     'nap', 'sodium', 'fat', 'carbohydrates', 'protein', 'fibre', 'kcal_in', 'M_illumination'], axis=1)

# drop days without mood rating
for day, _ in df.iterrows():
    if df[target_label][day] != df[target_label][day]:  # checks for NaN
        df = df.drop(day)

print('df after dropped days without target', df)

# fill missing values with mean value
# get mean value
mean = df.agg(['mean'], axis=0)
for attribute_name in df.columns:
    nan_data_true_false = pd.isnull(df[attribute_name])
    nan_numeric_indices = pd.isnull(df[attribute_name]).to_numpy().nonzero()[0]
    nan_dates = nan_data_true_false[nan_numeric_indices].index
    for nan_date in nan_dates:
        substitute = mean[attribute_name][0]
        df.at[nan_date, attribute_name] = substitute

print('df after after missing substitution', df)

# dataframes to tensors
batch_size = 16
target_tensor = torch.tensor(df[target_label].values.astype(np.float32))
target_tensor = torch.unsqueeze(target_tensor, 1)  # due to one dim target tensor
# print('train_target', train_target)
input_df = df.drop([target_label], axis=1)
num_features = len(input_df.columns)
input_tensor = torch.tensor(input_df.values.astype(np.float32))
tensorDataset = data_utils.TensorDataset(input_tensor, target_tensor)

# train test split
train_size = int(0.9 * len(tensorDataset))
test_size = len(tensorDataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(tensorDataset, [train_size, test_size])

# load data
train_dataloader = data_utils.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = data_utils.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

for X, y in test_dataset:
    print("Shape of X [BatchSize, #params]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU(),
            # nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print('batch * len(X)',batch,len(X))
            loss, current = loss.item(), batch * len(X)
            print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Avg test loss: {test_loss:>8f} \n")


epochs = 500
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
