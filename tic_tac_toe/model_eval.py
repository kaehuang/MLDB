import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from joblib import load

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

# Load and preprocess the validation data
data_val = pd.read_csv('tic_tac_toe.csv')

# Convert the 'String' column into a DataFrame
board_df_val = data_val['String'].apply(list).apply(pd.Series)
board_df_val.columns = [f"pos_{i}" for i in range(1, 10)]

# One-hot encode the board strings
categories = [['x', 'o', '-']] * 9
encoder = load('encoder.joblib')
encoded_board_val = encoder.transform(board_df_val)
X_val = torch.tensor(encoded_board_val, dtype=torch.float32)

# Convert the 'Value' column to integers: Win=0, Tie=1, Lose=2
value_map = {'Win': 0, 'Tie': 1, 'Lose': 2}
y_value_val = data_val['Value'].map(value_map).values
y_value_val = torch.tensor(y_value_val, dtype=torch.int64)

# Get remoteness values
y_remoteness_val = torch.tensor(data_val['Remoteness'].values, dtype=torch.int64)  # Ensure consistent datatype

# Load the model
model = TicTacToeNet()
model.load_state_dict(torch.load('tic_tac_toe_model.pth'))
model.eval()

# Evaluation
with torch.no_grad():
    value_preds, remoteness_preds = model(X_val)

# Calculate accuracy for 'Value'
_, predicted = torch.max(value_preds, 1)
accuracy = (predicted == y_value_val).sum().item() / y_value_val.size(0)

# Calculate mean squared error for 'Remoteness'
mse = ((remoteness_preds.squeeze() - y_remoteness_val) ** 2).mean().item()

print(f"Accuracy for 'Value': {accuracy * 100:.2f}%")
print(f"Mean Squared Error for 'Remoteness': {mse:.4f}")