import os
from collections import Counter

import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

import configurations
from t4sa.t4sa_dataset import T4saDataset
from text_analyzer.model import SentimentLSTM, train_on_gpu
from text_analyzer.vocab import Vocab


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


def train2(vocab=None):
    if vocab is None:
        vocab = get_vocabulary()

    batch_size = 50

    train_dataset = T4saDataset(train=True, configs=configurations, load_image=False, limit=1000)
    test_dataset = T4saDataset(train=False, configs=configurations, load_image=False, limit=1000)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=train_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    output_size = 3
    embedding_dim = 200
    seq_length = 200
    hidden_dim = 256
    n_layers = 2
    vocab_size = len(vocab.vocab) + 1  # +1 for the 0 padding
    net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)
    # loss and optimization functions
    lr = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params

    epochs = 15  # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 2
    clip = 5  # gradient clipping

    # move model to GPU, if available
    if (train_on_gpu):
        net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for data in train_loader:
            counter += 1
            inputs, labels = data["description"], data["classification"]
            inputs = torch.from_numpy(vocab.encode(data["description"], seq_length))

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            #            inputs = inputs.type(torch.LongTensor)

            output, h = net.forward(inputs, h)
            print(output)


            print(output.squeeze())
            print(labels.float())
            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for data in test_loader:
                    inputs, labels = data["description"], data["classification"]
                    inputs = torch.from_numpy(vocab.encode(data["description"], seq_length))
                    labels = torch.from_numpy(np.asarray([np.asarray([element]) for element in labels]))

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net.forward(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))


train2()