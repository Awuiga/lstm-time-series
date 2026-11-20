import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data/temperatures.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop = True)

train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)

train_data = data.iloc[ : train_size]
val_data = data.iloc[train_size : train_size + val_size]
test_data = data.iloc[train_size + val_size :]

mean = train_data['Temp'].mean()
std = train_data['Temp'].std()
train_values = (train_data['Temp'] - mean) / std
val_values = (val_data['Temp'] - mean) / std
test_values = (test_data['Temp'] - mean) / std

train_values = torch.tensor(train_values.to_numpy(), dtype = torch.float32)
val_values = torch.tensor(val_values.to_numpy(), dtype = torch.float32)
test_values = torch.tensor(test_values.to_numpy(), dtype = torch.float32)

window_size = 30
inputs = train_values[:window_size]
targets = train_values[window_size]

class TimeSeriesDataset(Dataset):
    def __init__(self,values, window_size):
        self.values = values
        self.window_size = window_size

    def __len__(self):
        return len(self.values) - self.window_size

    def __getitem__(self, index):
        inputs = self.values[index : self.window_size + index]
        targets = self.values[self.window_size + index]

        inputs = inputs.unsqueeze(1)
        targets = targets.unsqueeze(-1)
        return inputs, targets

batch_size = 64

train_dataset = TimeSeriesDataset(train_values, window_size)
val_dataset = TimeSeriesDataset(val_values, window_size)
test_dataset = TimeSeriesDataset(test_values, window_size)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

inputs, targets = next(iter(train_dataloader))

input_size = 1
hidden_size = 128

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

model = TimeSeriesModel(input_size, hidden_size)
model = model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)
predictions = model(inputs)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)

loss = criterion(predictions, targets)

epochs = 30
best_val_loss = float('inf')


for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        train_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()
    avg_loss = train_loss / len(train_dataset)


    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            val_loss += loss.item() * inputs.size(0)
        avg_val_loss = val_loss / len(val_dataset)

        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'lstm_model.pt')
        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

model.load_state_dict(torch.load('lstm_model.pt', map_location=device))
model.eval()

test_loss = 0
test_mae = 0
baseline_loss = 0
baseline_mae = 0

all_predictions = []
all_targets = []
all_baseline_predictions = []

with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)
        baseline_predictions = inputs[:, -1, :]
        errors = predictions - targets
        baseline_errors = baseline_predictions - targets
        abs_loss = torch.abs(errors)
        test_mae += abs_loss.sum().item()
        baseline_batch_loss = criterion(baseline_predictions, targets)
        loss = criterion(predictions, targets)
        test_loss += loss.item() * inputs.size(0)
        baseline_loss += baseline_batch_loss.item() * inputs.size(0)
        baseline_abs_loss = torch.abs(baseline_errors)
        baseline_mae += baseline_abs_loss.sum().item()
        all_predictions.extend(predictions.cpu().squeeze(-1).tolist())
        all_targets.extend(targets.cpu().squeeze(-1).tolist())
        all_baseline_predictions.extend(baseline_predictions.cpu().squeeze(-1).tolist())
    avg_test_loss = test_loss / len(test_dataset)
    avg_baseline_loss = baseline_loss / len(test_dataset)
    rmse = (avg_test_loss ** 0.5) * std
    baseline_rmse = (avg_baseline_loss ** 0.5) * std
    mae = (test_mae / len(test_dataset)) * std
    baseline_mae_celsius = (baseline_mae / len(test_dataset)) * std
    all_predictions = np.array(all_predictions) * std + mean
    all_targets = np.array(all_targets) * std + mean
    all_baseline_predictions = np.array(all_baseline_predictions) * std + mean

    print(all_predictions[:5])
    print(all_targets[:5])
    print(all_baseline_predictions[:5])
    print(f'Test Loss: {avg_test_loss:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}')
    print(f'Baseline Loss: {avg_baseline_loss:.4f} | RMSE: {baseline_rmse:.4f} | MAE: {baseline_mae_celsius:.4f}')

plt.figure(figsize = (14, 6))
plt.plot(all_targets[:100], label = 'Actual')
plt.plot(all_predictions[:100], label = 'LSTM')
plt.plot(all_baseline_predictions[:100], label = 'Baseline')

plt.title('LSTM Temperature Prediction')
plt.xlabel('Time step')
plt.ylabel('Temperature (C)')
plt.legend()
plt.grid(True)

plt.savefig('temperature_predictions.png', dpi = 150)
plt.show()

check_point = {
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'window_size': window_size,
    'mean': float(mean),
    'std': float(std)
}

torch.save(check_point, 'check_point.pt')
