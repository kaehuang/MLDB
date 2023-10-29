import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder

# Load and preprocess the data
data = pd.read_csv('tic_tac_toe.csv')

# Convert the 'String' column into a DataFrame
board_df = data['String'].apply(list).apply(pd.Series)
board_df.columns = [f"pos_{i}" for i in range(1, 10)]

# One-hot encode the board strings
categories = [['x', 'o', '-']] * 9
encoder = OneHotEncoder(sparse=False, categories=categories)
encoded_board = encoder.fit_transform(board_df)
X = torch.tensor(encoded_board, dtype=torch.float32)

# Convert the 'Value' column to integers: Win=0, Tie=1, Lose=2
value_map = {'Win': 0, 'Tie': 1, 'Lose': 2}
y_value = data['Value'].map(value_map).values
y_value = torch.tensor(y_value, dtype=torch.int64)

# Get remoteness values as integers
y_remoteness = torch.tensor(data['Remoteness'].values, dtype=torch.int64)

# Define the Neural Network architecture using PyTorch
class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.hidden1 = nn.Linear(27, 18)
        self.hidden2 = nn.Linear(18, 9)
        self.value_output = nn.Linear(9, 3)
        self.remoteness_output = nn.Linear(9, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        value = self.value_output(x)
        remoteness = self.remoteness_output(x)
        return value, remoteness

# Custom Poisson loss
def poisson_loss(output, target):
    return torch.exp(output) - target * output

# Hyperparameters
epochs = 50000
lr = 0.001

# Define the model, loss, and optimizer
model = TicTacToeNet()
criterion_value = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

from joblib import dump

# ... [rest of the script]
dump(encoder, 'encoder.joblib')


# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    
    value_preds, remoteness_preds = model(X)
    
    loss_value = criterion_value(value_preds, y_value)
    loss_remoteness = poisson_loss(remoteness_preds.squeeze(), y_remoteness.float()).mean()
    
    # Combined loss
    loss = loss_value + loss_remoteness
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:  # Print every 10 epochs
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
torch.save(model.state_dict(), 'tic_tac_toe_model.pth')

print("Training complete and model saved.")
