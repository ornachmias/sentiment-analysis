import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim, batch_size):
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

        # Initialize hidden state with zeros
        self._lstm_hidden_state = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        # Initialize cell state
        self._lstm_cell_state = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()

    def forward(self, x):
        embeddings = self.word_embeddings(x)
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch

        out, (hn, cn) = self.lstm(embeddings, (self._lstm_hidden_state.detach(), self._lstm_cell_state.detach()))
        self._lstm_hidden_state = hn
        self._lstm_cell_state = cn

        # Index hidden state of last time step
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])

        soft_max = self.soft_max(out)
        return soft_max
