# Multi-Output Neural Network for Tic-Tac-Toe Analysis

## Overview
This document describes the architecture and training process of a multi-output neural network model designed for Tic-Tac-Toe game analysis. The model predicts both the game outcome (Win, Tie, Lose) and a remoteness score for each game state.

## Data Processing
- **Data Source**: The model is trained on data from a 'tic_tac_toe.csv' file.
- **Feature Encoding**:
  - The 'String' column, representing Tic-Tac-Toe board states, is converted into a DataFrame with individual positions (pos_1 to pos_9).
  - Each board position is one-hot encoded with three categories: 'x', 'o', '-', representing player moves and empty spaces.
- **Target Encoding**:
  - The 'Value' column, indicating game outcomes, is encoded as integers (Win: 0, Tie: 1, Lose: 2).
  - The 'Remoteness' column is treated as an integer value representing the remoteness score.

## Model Architecture
- **Class**: `TicTacToeNet`, a subclass of `torch.nn.Module`.
- **Layers**:
  - Two hidden layers with 18 and 9 neurons, respectively.
  - A value output layer with 3 neurons for game outcome prediction.
  - A remoteness output layer with 1 neuron for remoteness score prediction.
- **Activation**: ReLU activation function is used in hidden layers.

## Training Process
- **Loss Functions**:
  - CrossEntropyLoss for the value (game outcome) predictions.
  - Custom Poisson loss for the remoteness predictions.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Epochs**: The model is trained for 5000 epochs.
- **Combined Loss**: The model optimizes a combined loss function comprising both value and remoteness loss components.

## Model Saving
- The model's state dictionary is saved to 'tic_tac_toe_model.pth'.
- The one-hot encoder is serialized and saved to 'encoder.joblib'.

## Challenges and Observations
- The model is designed to perform two distinct tasks: classify the game outcome and regress the remoteness score.
- The remoteness prediction task might require additional refinement or a different approach, as its accuracy is not as high as the value prediction.

## Conclusion
This multi-output neural network model offers a novel approach to analyze Tic-Tac-Toe games, predicting both the game outcome and a numerical remoteness score. Further improvements might be needed for the remoteness prediction component.

