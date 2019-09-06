import torch

# Device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 5
batch_size = 50

# Architecture
num_classes = 2