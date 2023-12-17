import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from joblib import dump

# Load and preprocess the data
data = pd.read_csv('dao.csv')

# Convert the 'String' column into a DataFrame
# Split into 16 board positions and 1 turn indicator
board_df = data['String'].apply(lambda s: list(s[:16]) + [s[16:]]).apply(pd.Series)
board_df.columns = [f"pos_{i}" for i in range(1, 18)]

# One-hot encode the board strings
categories = [['X', 'O', '-']] * 16 + [['_X', '_O']]
encoder = OneHotEncoder(sparse=False, categories=categories)
encoded_board = encoder.fit_transform(board_df)
X = torch.tensor(encoded_board, dtype=torch.float32)

# Convert the 'Value' column to integers
value_map = {'Win': 0, 'Draw': 1, 'Lose': 2}
y_value = data['Value'].map(value_map).values
y_value = torch.tensor(y_value, dtype=torch.int64)

# Get remoteness values as integers
y_remoteness = torch.tensor(data['Remoteness'].values, dtype=torch.int64)

# Neural Network architecture
class DaoNet(nn.Module):
    def __init__(self):
        super(DaoNet, self).__init__()
        self.hidden1 = nn.Linear(50, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.value_output = nn.Linear(16, 3)
        self.remoteness_output = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        value = self.value_output(x)
        remoteness = self.remoteness_output(x)
        return value, remoteness

# Custom MSE loss for remoteness
def mse_loss(output, target):
    return ((output - target) ** 2).mean()

# Hyperparameters
epochs = 100
lr = 0.001

# Model, loss, and optimizer
model = DaoNet()
criterion_value = nn.CrossEntropyLoss()
criterion_remoteness = mse_loss
optimizer = optim.Adam(model.parameters(), lr=lr)

# Serialize and save the encoder
dump(encoder, 'dao_encoder.joblib')

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    
    value_preds, remoteness_preds = model(X)
    
    loss_value = criterion_value(value_preds, y_value)
    loss_remoteness = criterion_remoteness(remoteness_preds.squeeze(), y_remoteness.float())
    
    # Combined loss with a possible scaling factor
    loss = loss_value + loss_remoteness
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        # Calculate accuracy for 'Value' predictions
        _, predicted_values = torch.max(value_preds, 1)
        accuracy = accuracy_score(y_value.numpy(), predicted_values.numpy())
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}")

torch.save(model.state_dict(), 'dao_model.pth')
print("Training complete and model saved.")