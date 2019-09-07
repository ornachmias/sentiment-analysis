import torch
from torch import nn


# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
from configurations import DEVICE


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x):
        embeddings = self.word_embeddings(x)
        # Initialize hidden state with zeros
        lstm_hidden_state = torch.zeros(self.layer_dim, x.size()[0], self.hidden_dim).requires_grad_().to(DEVICE)
        # Initialize cell state
        lstm_cell_state = torch.zeros(self.layer_dim, x.size()[0], self.hidden_dim).requires_grad_().to(DEVICE)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch

        out, (hn, cn) = self.lstm(embeddings, (lstm_hidden_state.detach(), lstm_cell_state.detach()))

        # Index hidden state of last time step
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])

        soft_max = self.soft_max(out)
        return soft_max
