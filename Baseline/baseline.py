import torch.nn as nn
import torch.nn.functional as F
import torch

class Baseline(nn.Module):
    """
    Simple baseline lstm model
    """

    def __init__(self, time_steps, hidden_dim, num_classes):

        super(Baseline, self).__init__()
        self.hidden_dim = hidden_dim

        #lstm layer
        self.lstm = nn.LSTM(time_steps, hidden_dim)

        #output layer
        self.linear = nn.Linear(hidden_dim, num_classes)


    def forward(self, input):
        lstm_out, _ = self.lstm(input.view(len(input), 1, -1))
        predictions = self.linear(lstm_out.view(len(input), -1))
        return predictions[-1].view(1,-1)
