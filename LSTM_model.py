import os
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

class FloodLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FloodLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).float()  # Initialize hidden state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).float()  # Initialize cell state
        out, _ = self.lstm(x.float(), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def NSE(y_pred, y_true):
    mean_observed = torch.mean(y_true)
    numerator = torch.sum((y_true - y_pred) ** 2)
    denominator = torch.sum((y_true - mean_observed) ** 2)
    nse = 1 - (numerator / denominator)
    return nse


def persistNSE(y_pred, y_true, s_b):
    persistence_pred = torch.cat((y_true[0].unsqueeze(0), y_true[:-1]), dim=0)
    numerator = torch.sum((y_true - y_pred) ** 2)
    denominator = torch.sum((y_true - persistence_pred) ** 2)
    persist_nse = 1 - (numerator / denominator)
    return persist_nse


def KGE(y_pred, y_true):
    # Calculate Pearson correlation coefficient
    r = pearsonr(y_pred.detach().numpy(), y_true.detach().numpy())[0]

    # Calculate alpha and beta
    alpha = torch.std(y_pred) / torch.std(y_true)
    beta = torch.mean(y_pred) / torch.mean(y_true)

    # Calculate KGE
    kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge


def NSE_tag(y_pred, y_true, s_b, eps=1e-5):
    """
    We are creating a model for each of the gauge stations, se this function will be used to calculate the loss function
    for one of the stations at a time. The loss fonction is the NSE function, which is defined as:
    NSE = sum(((q(n*dt) - o(n*dt))^2) / ((s_b - eps)^2)) where n goes from 1 to Nb
    :param Nb: the number of time steps in this iteration
    :param y_pred: the simulated flow
    :param y_true: the observed flow
    :param s_b: the std of the observed flow, based on the training data
    :param eps: a small number to avoid division by zero
    :return:
    """
    numerator = torch.sum((y_pred - y_true) ** 2)
    denominator = (s_b - eps) ** 2
    return numerator / denominator

def train_loop(model, train_loader, optimizer, loss_fn):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    return train_loss / len(train_loader)

def val_loop(model, valid_loader, loss_fn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
      for data, targets in valid_loader:
          outputs = model(data)
          loss = loss_fn(outputs, targets)
          val_loss += loss.item()

    return val_loss / len(valid_loader)

def train_LSTM_model(model, train_loader, valid_loader, optimizer, loss_fn, num_epochs):
    all_train_losses, all_valid_losses = [], []
    for epoch in range(num_epochs):
        train_loss = train_loop(model, train_loader, optimizer, loss_fn)
        all_train_losses.append(train_loss)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss))
        val_loss = val_loop(model, valid_loader, loss_fn)
        all_valid_losses.append(val_loss)
        print(f'Validation Loss: {val_loss}')
    return all_train_losses, all_valid_losses


# Function to perform grid search for the hidden size of the LSTM network
# def calibrating_LSTM_model(train_data, train_labels, valid_data, valid_labels, loss_fn, hidden_sizes=[16, 32, 64, 128]):
#     best_loss = float('inf')
#     best_hidden_size = None
#
#     for hidden_size in hidden_sizes:
#         model = flood_LSTM(input_size=train_data.shape[-1], hidden_size=hidden_size, output_size=1)
#         train_LSTM_model(model, train_data, train_labels, loss_fn)
#         model.eval()
#         with torch.no_grad():
#             outputs = model(valid_data)
#             loss = loss_fn(outputs, valid_labels).item()
#
#         if loss < best_loss:
#             best_loss = loss
#             best_hidden_size = hidden_size
#
#     return best_hidden_size





class FloodDataset(Dataset):
    def __init__(self, data, sequence_length=144, forecast_length=24):
        self.data = data[:, :-1]
        self.labels = data[:, -1]
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.data) - self.sequence_length - self.forecast_length + 1

    def __getitem__(self, idx):
        # Get the label for the current index
        label = self.labels[idx + self.sequence_length : idx + self.sequence_length + self.forecast_length]

        # Get the previous sequence_length samples as the sequence
        sequence = self.data[idx : idx + self.sequence_length]

        return sequence, label


if __name__ == '__main__':
    # we want to create an LSTM model for each gauge station, based on the data we have in the dfs_per_gauge folder
    # we will create a model for each gauge station and save it in the models folder
    label_col = 'flow_rate'
    train_portion = 0.8
    hours_to_forcast = 2
    hours_to_base_on = 24
    hidden_size = 64  # Number of hidden units in LSTM layer
    num_layers = 2  # Number of LSTM layers
    output_size = forecast_length = hours_to_forcast * 6  # Number of time steps to forecast
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10

    gauge_df = pd.read_csv('dfs_per_gauge_full_data/7105.csv')
    gauge_df = gauge_df.replace('-', np.nan)
    gauge_df = gauge_df[~gauge_df['flow_rate'].isna()]
    gauge_df = gauge_df[(~gauge_df['ims_1_rain'].isna()) & (~gauge_df['ims_2_rain'].isna()) & (~gauge_df['ims_3_rain'].isna())& (~gauge_df['ims_4_rain'].isna())& (~gauge_df['ims_5_rain'].isna())]
    gauge_df = gauge_df.loc[gauge_df.apply(pd.Series.first_valid_index).max(): gauge_df.apply(pd.Series.last_valid_index).min()].reset_index(drop=True)
    gauge_df = gauge_df.drop(columns=['datetime', 'gauge_id', 'ims_1_id', 'ims_2_id', 'ims_3_id', 'ims_4_id', 'ims_5_id'])
    gauge_df[['ims_1_dist', 'ims_2_dist', 'ims_3_dist', 'ims_4_dist', 'ims_5_dist']] *= 1000

    train_features, train_targets = gauge_df.drop(columns=[label_col]), gauge_df[label_col]
    train_data = np.hstack((train_features.values, train_targets.values.reshape(-1, 1))).astype(np.float32)
    # divide the data into training and validation data
    train_data, valid_data = train_data[:int(len(train_data) * train_portion)], train_data[
                                                                                int(len(train_data) * train_portion):]
    train_data, valid_data = torch.tensor(train_data).float(), torch.tensor(valid_data).float()
    train_targets, valid_targets = train_targets[:int(len(train_targets) * train_portion)], train_targets[int(len(
        train_targets) * train_portion):]
    train_targets, valid_targets = torch.tensor(train_targets.values).float(), torch.tensor(
        valid_targets.values).float()

    train_loader = DataLoader(
        FloodDataset(train_data, sequence_length=hours_to_base_on * 6, forecast_length=forecast_length),
        batch_size=batch_size)
    valid_loader = DataLoader(
        FloodDataset(valid_data, sequence_length=hours_to_base_on * 6, forecast_length=forecast_length),
        batch_size=batch_size)

    model = FloodLSTM(train_features.shape[1], hidden_size, num_layers, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    s_b = train_targets.std().item()
    nse_tag_loss = lambda y_pred, y_true: NSE_tag(y_pred, y_true, s_b)

    all_train_losses, all_valid_losses = train_LSTM_model(model, train_loader, valid_loader, optimizer, nse_tag_loss,
                                                          num_epochs)

    # torch.save(model.state_dict(), 'models/7105_model.pth')
    plt.plot(all_train_losses, label='Training Loss')
    plt.plot(all_valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss using NSE_tag')
    plt.show()

