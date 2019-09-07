import os

import torch

data_path = os.path.join(os.path.dirname(__file__), "data")
images_directory = os.path.join(data_path, "b-t4sa_imgs")
images_classification_train_file = os.path.join(images_directory, "b-t4sa_train.txt")
images_classification_eval_file = os.path.join(images_directory, "b-t4sa_val.txt")
descriptions_file = os.path.join(data_path, "raw_tweets_text.csv")
logs_dir = "./logs"
image_size = 224
batch_size = 20
training_size = 1000
eval_size = 100
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")