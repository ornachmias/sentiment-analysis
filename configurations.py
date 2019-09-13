import os

import torch

# Paths
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
images_directory = os.path.join(data_path, "b-t4sa_imgs")
images_classification_train_file = os.path.join(images_directory, "b-t4sa_train.txt")
images_classification_eval_file = os.path.join(images_directory, "b-t4sa_val.txt")
descriptions_file = os.path.join(data_path, "raw_tweets_text.csv")

# Training and network parameters
image_size = 224
batch_size = 25
training_size = 50000  # Change to None if we want to run on all training
eval_size = 1000  # Change to None if we want to run on all training
output_size = 2
epochs = 5
clip = 5
seq_length = 150
embedding_dim = 150

# Run parameters
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
