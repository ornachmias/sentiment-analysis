import torch

from image_sentiment.vgg19 import settings, train
from image_sentiment.vgg19.model import Model
from t4sa.t4sa_dataset import T4saDataset
from torch.utils.data import DataLoader

import configurations


configurations.images_directory = "../../data/b-t4sa_imgs"
configurations.images_classification_train_file = "../../data/b-t4sa_imgs/b-t4sa_train.txt"
configurations.images_classification_eval_file = "../../data/b-t4sa_imgs/b-t4sa_val.txt"
configurations.descriptions_file = "../../data/raw_tweets_text.csv"


train_dataset = T4saDataset(train=True, configs=configurations, load_image=True, limit=5000)
test_dataset = T4saDataset(train=False, configs=configurations, load_image=True, limit=1000)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=settings.batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=settings.batch_size,
                         shuffle=False)

for features in train_loader:
    print('Image batch dimensions:', features["image"].shape)
    print('Image label dimensions:', features["classification"].shape)
    break

torch.manual_seed(settings.random_seed)
model = Model(num_classes=settings.num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
train.run(model, train_loader, optimizer)