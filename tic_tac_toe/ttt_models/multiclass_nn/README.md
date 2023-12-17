# Tic-Tac-Toe Neural Network Models Overview

## General Observation
- Models with only one linear layer demonstrate approximately 47-48% accuracy.

## Model Summaries

### Model: tttModelV1.pth
- **Size**: 14.7kB
- **Accuracy**: 98.99%
- **Training Parameters**:
  - Learning Rate: 0.02
  - Epochs: 7000
- **Layers**:
  - `nn.Linear(9, 128)`
  - `nn.ReLU()`
  - `nn.Linear(128, 16)`

### Model: tttModelV2.pth
- **Size**: 9.91kB
- **Accuracy**: 83.69%
- **Training Parameters**:
  - Learning Rate: 0.05
  - Epochs: 10000
- **Layers**:
  - `nn.Linear(9, 80)`
  - `nn.ReLU()`
  - `nn.Linear(80, 16)`

### Model: tttModelV3.pth
- **Size**: 8.28kB
- **Accuracy**: 90.47%
- **Training Parameters**:
  - Learning Rate: 0.05
  - Epochs: 10000
- **Layers**:
  - `nn.Linear(9, 64)`
  - `nn.Sigmoid()`
  - `nn.Linear(64, 16)`

### Model: tttModelV4.pth
- **Size**: 6.66kB
- **Accuracy**: 81.03%
- **Training Parameters**:
  - Learning Rate: 0.03
  - Epochs: 20000
- **Layers**:
  - `nn.Linear(9, 48)`
  - `nn.Sigmoid()`
  - `nn.Linear(48, 16)`

### Model: tttModelV5.pth (Basic Model)
- **Size**: 1.72kB
- **Accuracy**: 48.03%
- **Training Parameters**:
  - Learning Rate: 0.01
  - Epochs: 10000
- **Layers**:
  - `nn.Linear(9, 16)`

### Model: tttHashModelV1.pth (Basic Model with Hash Values)
- **Size**: 1.49kB
- **Accuracy**: 44.19%
- **Training Parameters**:
  - Learning Rate: 0.0005
  - Epochs: 199 (terminated manually)
- **Layers**:
  - `nn.Linear(5, 16)`
