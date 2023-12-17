# DaoNet: A Hybrid Neural Network for the Dao Game

## Overview
This document describes the architecture, training, and functionality of `DaoNet`, a neural network model developed for analyzing and predicting outcomes in the Dao board game. The model is designed to perform both classification (game outcome) and regression (remoteness score) tasks. We are utilizing a larger game dataset to see if we can use ML to improve database compression.

## Data Preparation and Encoding
- **Dataset**: The model is trained on data from a file named 'dao.csv'.
- **Board State Encoding**:
  - The 'String' column, representing the game board, is split into 16 board positions and 1 turn indicator.
  - One-hot encoding is applied to these positions and the turn indicator.
- **Target Encoding**:
  - The 'Value' column, indicating game outcomes (Win, Draw, Lose), is mapped to integers.
  - The 'Remoteness' column is treated as a numerical value.

## Model Architecture
- **Class**: `DaoNet`, a subclass of `torch.nn.Module`.
- **Layers**:
  - Two hidden layers with 32 and 16 neurons, respectively, using ReLU activations.
  - A value output layer with 3 neurons for game outcome prediction.
  - A remoteness output layer with 1 neuron for remoteness score prediction.

## Loss Function and Optimization
- **Loss Functions**:
  - CrossEntropyLoss for the value (game outcome) predictions.
  - Mean Squared Error (MSE) loss for the remoteness predictions.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.

## Training Process
- **Epochs**: The model is trained for 100 epochs.
- **Loss and Accuracy**:
  - The model optimizes a combined loss comprising both value and remoteness loss components.
  - Accuracy for the value predictions is calculated each epoch to monitor classification performance.

## Serialization and Storage
- The one-hot encoder used for preprocessing is saved as 'dao_encoder.joblib'.
- The trained model's state dictionary is saved as 'dao_model.pth'.

## Application and Purpose
- `DaoNet` is capable of predicting both the categorical game outcome (Win, Draw, Lose) and a numerical remoteness score from a given game state.
- This model can be instrumental in game strategy analysis, AI development for gameplay, or outcome prediction in the Dao game.

## Conclusion
`DaoNet` represents an innovative approach to game analysis, combining classification and regression tasks within a single neural network model, tailored specifically for the Dao game.
