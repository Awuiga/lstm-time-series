import torch
import torch.nn as nn
import pandas as pd



class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        output, (hidden, cell) = self.lstm(inputs)
        last_hidden = output[:, -1, :]
        predictions = self.regressor(last_hidden)
        return predictions

device = 'cuda' if torch.cuda.is_available() else 'cpu'
check_point = torch.load(
    'check_point.pt',
    map_location = device,
    weights_only = True
)

input_size = check_point['input_size']
hidden_size = check_point['hidden_size']
window_size = check_point['window_size']
mean = check_point['mean']
std = check_point['std']

model = TimeSeriesModel(input_size, hidden_size)
model = model.to(device)
model.load_state_dict(check_point['model_state_dict'])
model.eval()

data = pd.read_csv('data/temperatures.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop = True)
inputs = data['Temp'].tail(window_size)
inputs = (inputs  - mean) / std

inputs = torch.tensor(inputs.to_numpy(), dtype = torch.float32)
inputs = inputs.unsqueeze(0)
inputs = inputs.unsqueeze(-1)
inputs = inputs.to(device)

with torch.no_grad():
    predictions = model(inputs)
    predictions = predictions.item()
    predictions = (predictions * std) + mean
print(predictions)
