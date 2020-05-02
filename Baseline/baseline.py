import torch.nn as nn
import torch

class Baseline(nn.Module):
    """
    Simple baseline lstm model
    """

    def __init__(self, time_steps, prediction_length):

        super(Baseline, self).__init__()
        self.prediction_length = prediction_length

        #lstm layer
        self.lstm = nn.LSTM(time_steps, prediction_length)
        self.hidden_cell = (torch.zeros(1,1,self.prediction_length), torch.zeros(1,1,self.prediction_length))

        #output layer
        self.linear = nn.Linear(prediction_length, prediction_length)


    def forward(self, input):
        lstm_out, self.hidden_cell = self.lstm(input.view(len(input) ,1 , -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input), -1))
        return predictions[-1]
