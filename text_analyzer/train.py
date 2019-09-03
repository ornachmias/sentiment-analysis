import os

import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

import configurations
from t4sa.t4sa_dataset import T4saDataset
from text_analyzer import settings
from text_analyzer.model import LSTMModel
from text_analyzer.vocab import Vocab

train_on_gpu = False


def get_vocabulary(refresh=False):
    vocab_path = os.path.join("vocabulary.pickle")
    if os.path.exists(vocab_path) and refresh:
        os.remove(vocab_path)
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f_obj:
            return pickle.load(f_obj)

    train_dataset = T4saDataset(train=True, configs=configurations, load_image=False)
    texts = [item["description"] for item in train_dataset]
    vocab = Vocab(texts)
    with open(vocab_path, "wb") as f_obj:
        pickle.dump(vocab, f_obj)
    return vocab


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def get_train_loader():
    train_dataset = T4saDataset(train=True, configs=configurations, load_image=False, limit=50000)
    return DataLoader(dataset=train_dataset,
                      batch_size=settings.batch_size,
                      shuffle=True)


def get_test_loader():
    test_dataset = T4saDataset(train=False, configs=configurations, load_image=False, limit=100)
    return DataLoader(dataset=test_dataset,
                      batch_size=settings.batch_size,
                      shuffle=False)


def _train(net, vocab, train_loader, criterion, optimizer, epochs):
    net.train()
    # train for some number of epochs
    for e in range(epochs):
        step = 0
        # initialize hidden state

        # batch loop
        for data in train_loader:
            step += 1
            optimizer.zero_grad()

            inputs, labels = data["description"], data["classification"]
            labels = torch.from_numpy(indices_to_one_hot(labels, settings.output_size))
            labels = torch.tensor(labels, dtype=torch.float)
            inputs = torch.from_numpy(vocab.encode(data["description"], settings.seq_length))

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero accumulated gradients
            net.zero_grad()
            output = net.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), settings.clip)
            optimizer.step()

            yield (e, step)


def _evaluate(net, vocab, test_loader, criterion):
    # Get validation loss
    val_losses = []
    correct = 0
    total = 0
    net.eval()
    for data in test_loader:
        inputs, labels = data["description"], data["classification"]
        hotspot_labels = torch.from_numpy(indices_to_one_hot(labels, settings.output_size))
        hotspot_labels = torch.tensor(hotspot_labels, dtype=torch.float)
        inputs = torch.from_numpy(vocab.encode(data["description"], settings.seq_length))
        if (train_on_gpu):
            inputs, hotspot_labels = inputs.cuda(), labels.cuda()

        output = net.forward(inputs)
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
          "Accuracy : {:.6f}".format(accuracy)
          )


def train_and_evaluate():
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    vocab = get_vocabulary()
    vocab_size = len(vocab.vocab) + 1  # +1 for the 0 padding
    net = LSTMModel(vocab_size, settings.embedding_dim, settings.hidden_dim, settings.layer_dim, settings.output_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.lr)
    print_every = 5
    epochs = 2

    for (epoch, step) in _train(net, vocab, train_loader, criterion, optimizer, epochs):
        if step % print_every == 0:
            print("Epoch: {}/{}...".format(epoch + 1, epochs),
                  "Step: {}...".format(step))
            _evaluate(net, vocab, test_loader, criterion)

    return net


def get_model():
    path = os.path.join(os.path.dirname(__file__), "trained_sentence_sentiment_model")
    if os.path.exists(path):
        return torch.load(path)
    model = train_and_evaluate()
    torch.save(model, path)
    return model


get_model()
