import torch

# Device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 2
batch_size = 128

# Architecture
num_classes = 3