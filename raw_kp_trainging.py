import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pandas as pd
from getKpindex import getKpindex
import matplotlib.pyplot as plt
import csv


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Load dscovr_raw_clean.csv
print("Load Data!")
df_input = pd.read_csv('./Data/dscovr_raw_clean.csv')
df_input['Datetime'] = pd.to_datetime(df_input['Datetime']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
(time, index, status) = getKpindex(df_input['Datetime'].iloc[0], df_input['Datetime'].iloc[-1], 'Kp')
df_output = pd.DataFrame({'Datetime': time, 'Kp': index})

# Convert the 'Datetime' column to datetime objects in both DataFrames
df_input['Datetime'] = pd.to_datetime(df_input['Datetime'])
df_output['Datetime'] = pd.to_datetime(df_output['Datetime'])

merged_df = pd.merge_asof(df_input, df_output, on='Datetime', direction='backward')


n_epochs = 10000
batch_size = 100
sequence_length = 180 #180 minutes = 3 hours

#Data Preparation
print("Prepare Data!")
# Feature Selection and Normalization
selected_features = ['mfv_gse_x', 'mfv_gse_y', 'mfv_gse_z', 'Kp']
data = merged_df[selected_features].values
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
# Create Input Sequences and Targets
sequence_length = 180  # 180 minutes = 3 hours
X_input = []
y_output = []
for i in range(len(data) - sequence_length):
    X_input.append(data[i:i + sequence_length, :-1])  # Input: 'mfv_gse_x', 'mfv_gse_y', 'mfv_gse_z'
    y_output.append(data[i + sequence_length, -1])  # Output: 'Kp'
X_input = np.array(X_input)
y_output = np.array(y_output)

# #TODO Comment out later
# X_input = X_input[:50000]
# y_output = y_output[:50000]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_input, y_output, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.Tensor(X_train).to(device)
y_train = torch.Tensor(y_train).to(device)
X_test = torch.Tensor(X_test).to(device)
y_test = torch.Tensor(y_test).to(device)

# Create DataLoader for Training
train_dataset = TensorDataset(X_train, y_train)
train_sampler = SubsetRandomSampler(range(len(train_dataset)))
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

# Create DataLoader for Testing
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#LSTM Model
class DSTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=8, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(8, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

model = DSTModel()
model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()


#Training part
train_losses = []
test_losses = []
print("Train Model!")
total_train_batches = len(train_loader)
total_test_batches = len(test_loader)
stored_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    tot_loss = 0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = criterion(y_pred, torch.unsqueeze(y_batch, 1))
        tot_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if batch_idx % 100 == 0:
            progress_ratio = (batch_idx + 1) / total_train_batches
            print(f"Batch {batch_idx+1}/{total_train_batches} [{100 * progress_ratio:.2f}%] - Train Loss: {loss.item():.4f}")
    train_losses.append(tot_loss/total_train_batches)
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss += criterion(y_pred, torch.unsqueeze(y_batch, 1)).item()
            
            
            if batch_idx % 100 == 0:
                progress_ratio = (batch_idx + 1) / total_test_batches
                print(f"Batch {batch_idx+1}/{total_test_batches} [{100 * progress_ratio:.2f}%] - Test Loss: {loss:.4f}")
    test_losses.append(loss/total_test_batches)

    if loss < best_loss:
        best_loss = loss
        torch.save([model.state_dict()], "./Models/dst_kp_lstm_state_dict.pth")   
            
    print("epoch %d: loss: %.4f" % (epoch, loss))

    # Save loss values to a CSV file
    loss_data = {
        'epoch': epoch+1,
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1]
    }
    with open('./Data/loss_log.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'test_loss'])
        if epoch == 0:
            writer.writeheader()
        writer.writerow(loss_data)


# Plot and save the train and test loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Test Loss')
plt.grid(True)

# Show the plot
plt.show()