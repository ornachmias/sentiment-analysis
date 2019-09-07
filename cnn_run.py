import torch
import os

import numpy as np
from torch import nn
from torch.utils.data.dataloader import DataLoader

import configurations
from cnn_model import CnnModel
from t4sa_dataset import T4saDataset

# Hyperparameters
lr = 0.001


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def get_train_loader():
    train_dataset = T4saDataset(train=True, configs=configurations, load_image=True, limit=configurations.training_size)
    return DataLoader(dataset=train_dataset,
                      batch_size=configurations.batch_size,
                      shuffle=True)


def get_t_loader():
    test_dataset = T4saDataset(train=False, configs=configurations, load_image=True, limit=configurations.eval_size)
    return DataLoader(dataset=test_dataset,
                      batch_size=configurations.batch_size,
                      shuffle=False)


def _train(net, train_loader, criterion, optimizer, epochs):
    net.train()
    # train for some number of epochs
    for e in range(epochs):
        step = 0
        # initialize hidden state

        # batch loop
        for data in train_loader:
            step += 1
            optimizer.zero_grad()

            labels = data["classification"]
            images = data["image"].to(configurations.DEVICE)
            labels = torch.from_numpy(indices_to_one_hot(labels, configurations.output_size))
            labels = torch.tensor(labels, dtype=torch.float, device=configurations.DEVICE)

            # zero accumulated gradients
            net.zero_grad()
            output = net.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            yield (e, step)


def _evaluate(net, test_loader, criterion):
    # Get validation loss
    val_losses = []
    correct = 0
    total = 0
    net.eval()
    for data in test_loader:
        labels = data["classification"]
        images = data["image"].to(configurations.DEVICE)
        hotspot_labels = torch.from_numpy(indices_to_one_hot(labels, configurations.output_size))
        hotspot_labels = torch.tensor(hotspot_labels, dtype=torch.float, device=configurations.DEVICE)

        output = net.forward(images)
        loss = criterion(output, hotspot_labels)
        val_loss = criterion(output, hotspot_labels)
        val_losses.append(val_loss.item())

        _, predicted = torch.max(output.data, 1)
        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    net.train()
    print("Loss: {:.6f}...".format(loss.item()),
          "Val Loss: {:.6f}".format(np.mean(val_losses)),
          "Accuracy : {:.6f}".format(accuracy))


def train_and_evaluate():
    train_loader = get_train_loader()
    test_loader = get_t_loader()
    net = CnnModel().to(configurations.DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    print_every = 5

    for (epoch, step) in _train(net, train_loader, criterion, optimizer, configurations.epochs):
        if step % print_every == 0:
            print("Epoch: {}/{}...".format(epoch + 1, configurations.epochs),
                  "Step: {}...".format(step))
            _evaluate(net, test_loader, criterion)

    return net


def get_model():
    path = os.path.join(os.path.dirname(__file__), "trained_image_sentiment_model")
    if os.path.exists(path):
        return torch.load(path)
    model = train_and_evaluate()
    torch.save(model, path)
    return model


get_model()
