import math

import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from config import target_label

writer = SummaryWriter()

epochs = 1000
lr = 0.0001
torch.manual_seed(0)
weight_decay = 1


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, 16),
            # nn.ReLU(),
            # nn.Linear(num_features, 16),
            # nn.ReLU(),
            # nn.Linear(16, 8),
            # nn.ReLU(),
            # nn.Linear(8, 3),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def fully_connected_nn_prediction(df):
    # dataframes to tensors
    target_tensor = torch.tensor(df[target_label].values.astype(np.float32))
    target_tensor = torch.unsqueeze(target_tensor, 1)  # due to one dim target tensor
    # print('train_target', train_target)
    input_df = df.drop([target_label], axis=1)
    num_features = len(input_df.columns)
    input_tensor = torch.tensor(input_df.values.astype(np.float32))

    # input normalization
    scaler = MinMaxScaler()
    scaler.fit(input_tensor)
    input_tensor = torch.tensor(scaler.transform(input_tensor).astype(np.float32))

    tensor_dataset = data_utils.TensorDataset(input_tensor, target_tensor)

    # train test split
    print('dataset_size:', len(tensor_dataset))
    train_size = int(0.9 * len(tensor_dataset))
    test_size = len(tensor_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(tensor_dataset, [train_size, test_size])

    # load data
    batch_size = math.floor(train_size)
    train_dataloader = data_utils.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    for X, y in test_dataset:
        print("Shape of X [BatchSize, #params]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = NeuralNetwork(num_features).to(device)
    print(model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, epoch + 1, device)
        test(test_dataloader, model, loss_fn, epoch + 1, device)
    writer.flush()
    writer.close()
    print("Done Training!")

    # save model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # load model
    model = NeuralNetwork(num_features)
    model.load_state_dict(torch.load("model.pth"))

    model.eval()

    for day in range(len(test_dataset)):
        x = test_dataset[day][0]
        y = test_dataset[day][1]
        with torch.no_grad():
            pred = model(x)
            predicted, actual = pred[0], y
            print(f'Predicted: {predicted}; Actual: {actual[0]}')


def train(dataloader, model, loss_fn, optimizer, epoch, device):
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
            loss, current = loss.item(), batch * len(X)
            print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar("Loss/train", loss, epoch)


def test(dataloader, model, loss_fn, epoch, device):
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
    writer.add_scalar("Loss/test", test_loss, epoch)
