import torch

# Device
DEVICE = torch.device("cpu")

# Hyperparameters
batch_size = 100
output_size = 3
embedding_dim = 100
seq_length = 150
hidden_dim = 128
layer_dim = 3
lr = 0.01
clip = 5

# Architecture
num_classes = 3
