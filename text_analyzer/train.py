import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from log_writer import write_parameters, write_batch


import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

import configurations
from t4sa_dataset import T4saDataset
from text_analyzer.model import LSTMModel
from vocab import Vocab


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
    train_dataset = T4saDataset(train=True, configs=configurations, load_image=False, limit=configurations.training_size)
    return DataLoader(dataset=train_dataset,
                      batch_size=configurations.batch_size,
                      shuffle=True)


def get_test_loader():
    test_dataset = T4saDataset(train=False, configs=configurations, load_image=False, limit=configurations.eval_size)
    return DataLoader(dataset=test_dataset,
                      batch_size=configurations.batch_size,
                      shuffle=True)


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
            labels = torch.from_numpy(indices_to_one_hot(labels, configurations.output_size)).type(torch.float).to(
                configurations.DEVICE)
            inputs = torch.from_numpy(vocab.encode(data["description"], configurations.seq_length)).to(
                configurations.DEVICE)

            # zero accumulated gradients
            net.zero_grad()
            output = net.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), configurations.clip)
            optimizer.step()

            yield (e, step)


def _evaluate(net, vocab, test_loader, criterion, epoch, step):
    # Get validation loss
    val_losses = []
    correct = 0
    total = 0
    net.eval()
    for data in test_loader:
        inputs, labels = data["description"], data["classification"]
        hotspot_labels = torch.from_numpy(indices_to_one_hot(labels, configurations.output_size))
        hotspot_labels = torch.tensor(hotspot_labels, dtype=torch.float)
        inputs = torch.from_numpy(vocab.encode(data["description"], configurations.seq_length))
        inputs, hotspot_labels = inputs.to(configurations.DEVICE), hotspot_labels.to(configurations.DEVICE)

        output = net.forward(inputs)
        loss = criterion(output, hotspot_labels)
        val_loss = criterion(output, hotspot_labels)
        val_losses.append(val_loss.item())

        _, predicted = torch.max(output.data, 1)

        # Total number of labels
        total += labels.size(0)

        labels = labels.to(configurations.DEVICE)

        # Total correct predictions
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    net.train()
    write_batch(model_name="rnn", epoch=epoch, batch=step, accuracy=accuracy.item(), loss=loss.item())
    print("Loss: {:.6f}...".format(loss.item()),
          "Val Loss: {:.6f}".format(np.mean(val_losses)),
          "Accuracy : {:.6f}".format(accuracy))


def train_and_evaluate():
    hidden_dim = 128
    layer_dim = 3
    lr = 0.01

    train_loader = get_train_loader()
    test_loader = get_test_loader()
    vocab = get_vocabulary()
    vocab_size = len(vocab.vocab) + 1  # +1 for the 0 padding
    write_parameters("rnn", {"image_size": configurations.image_size,
                                 "batch_size": configurations.batch_size,
                                 "training_size": configurations.training_size,
                                 "eval_size": configurations.eval_size,
                                 "epochs": configurations.epochs,
                                 "output_size": configurations.output_size,
                                 "hidden_dim": hidden_dim,
                                 "embedding_dim": configurations.embedding_dim,
                                 "vocab_size": vocab_size,
                                 "layer_dim": layer_dim})
    net = LSTMModel(vocab_size, configurations.embedding_dim, hidden_dim, layer_dim, configurations.output_size).to(configurations.DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    print_every = 5
    epochs = configurations.epochs

    for (epoch, step) in _train(net, vocab, train_loader, criterion, optimizer, epochs):
        if step % print_every == 0:
            print("Epoch: {}/{}...".format(epoch + 1, epochs),
                  "Step: {}...".format(step))
            _evaluate(net, vocab, test_loader, criterion, epoch, step)

    return net


def run_t(net):
    criterion = nn.BCELoss()
    vocab = get_vocabulary()
    test_loader = get_test_loader()
    val_losses = []
    correct = 0
    total = 0
    net.eval()
    for data in test_loader:
        inputs, labels = data["description"], data["classification"]
        hotspot_labels = torch.from_numpy(indices_to_one_hot(labels, configurations.output_size))
        hotspot_labels = torch.tensor(hotspot_labels, dtype=torch.float)
        inputs = torch.from_numpy(vocab.encode(data["description"], configurations.seq_length))
        inputs, hotspot_labels = inputs.to(configurations.DEVICE), hotspot_labels.to(configurations.DEVICE)

        output = net.forward(inputs)
        loss = criterion(output, hotspot_labels)
        val_loss = criterion(output, hotspot_labels)
        val_losses.append(val_loss.item())

        _, predicted = torch.max(output.data, 1)

        # Total number of labels
        total += labels.size(0)

        labels = labels.to(configurations.DEVICE)

        # Total correct predictions
        correct += (predicted == labels).sum()

        accuracy = 100 * correct / total
        print("Loss: {:.6f}...".format(loss.item()),
              "Val Loss: {:.6f}".format(np.mean(val_losses)),
              "Accuracy : {:.6f}".format(accuracy))


def get_model():
    path = os.path.join(os.path.dirname(__file__), "trained_sentence_sentiment_model")
    if os.path.exists(path):
        return torch.load(path)
    model = train_and_evaluate()
    torch.save(model, path)
    return model


net = train_and_evaluate()
run_t(net)

