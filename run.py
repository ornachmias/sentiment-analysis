import torch

import settings
import train
from model import Model
from t4sa_dataset import T4saDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import configurations


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, features in enumerate(train_loader):
        data = features["image"]
        target = features["classification"]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for features in test_loader:
            data = features["image"]
            target = features["classification"]
            data, target = data.to(device), target.to(device)
            logits, output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



configurations.images_directory = "./data/b-t4sa_imgs"
configurations.images_classification_train_file = "./data/b-t4sa_imgs/b-t4sa_train.txt"
configurations.images_classification_eval_file = "./data/b-t4sa_imgs/b-t4sa_val.txt"
configurations.descriptions_file = "./data/raw_tweets_text.csv"

train_dataset = T4saDataset(train=True, configs=configurations, load_image=True, limit=500)
test_dataset = T4saDataset(train=False, configs=configurations, load_image=True, limit=50)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=settings.batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=settings.batch_size,
                         shuffle=False)

torch.manual_seed(settings.random_seed)
model = Model(num_classes=settings.num_classes).to(settings.DEVICE)

optimizer = torch.optim.RMSprop(model.parameters(), lr=settings.learning_rate)

for epoch in range(1, settings.num_epochs + 1):
    train(model, settings.DEVICE, train_loader, optimizer, epoch)
    test(model, settings.DEVICE, test_loader)

torch.save(model.state_dict(), "cnn.pt")