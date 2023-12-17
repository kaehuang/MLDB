# MLP Model for Tic-Tac-Toe (ttt)

## Overview
This document describes the Multi-Layer Perceptron (MLP) model designed for analyzing and predicting outcomes in Tic-Tac-Toe games. The model utilizes a unique approach by treating both game outcome prediction and remoteness score estimation as classification problems.

## Input
The model accepts input in the form of a feature vector, encoded based on the 'String' column from the Tic-Tac-Toe dataset. The encoding scheme is as follows:
- 'x': Encoded as 2
- 'o': Encoded as 1
- '-': Encoded as 0

## Model Architecture
The MLP model comprises two separate classification branches:
1. **Value Prediction Branch**: Predicts the game outcome (Win, Tie, Lose).
2. **Remoteness Prediction Branch**: Estimates the remoteness score of a game state, which is also treated as a classification task.

## Issues and Observations
- **Accuracy Concern**: The model exhibits a significant disparity in accuracy between the value prediction and the remoteness prediction. The accuracy of predicting the game outcome is notably higher compared to the accuracy of predicting the remoteness score.
- **Model Simplification**: Due to the relatively straightforward architecture, the serialized model file is compact, around 29KB in size. This compactness is beneficial for storage and deployment.
- **Exception Handling**: There is a notable presence of exceptions, particularly in the context of remoteness score predictions. This indicates potential areas for model improvement or a need for a more nuanced approach to handling the remoteness score.

## Conclusion
The MLP model presents an innovative approach to analyzing Tic-Tac-Toe games. However, it highlights the challenges in accurately predicting the remoteness score, suggesting the need for further refinement or a different modeling strategy for this particular aspect.
