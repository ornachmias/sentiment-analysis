import torch
from torch import nn


# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
class CombinedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        super(CombinedModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self._features_dim = 21632
        self.image_lstm = nn.LSTM(self._features_dim, hidden_dim, layer_dim, batch_first=True)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.soft_max = nn.Softmax(dim=1)


        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=0.2)
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=0.2)
        )

    def forward(self, sentences, images):
        images_out = self.features1(images)
        images_out = self.features2(images_out)
        images_out = self.features3(images_out)
        images_out = self.features4(images_out)
        images_out = images_out.view(-1, self._features_dim)

        embeddings = self.word_embeddings(sentences)
        # # Initialize hidden state with zeros
        # lstm_hidden_state = torch.zeros(self.layer_dim, sentences.size()[0], self.hidden_dim).requires_grad_()
        # # Initialize cell state
        #lstm_cell_state = torch.zeros(self.layer_dim, sentences.size()[0], self.hidden_dim).requires_grad_()

        images_lstm_hidden_state = torch.zeros(self.layer_dim, images_out.size()[0], self.hidden_dim).requires_grad_()
        images_lstm_cell_state = torch.zeros(self.layer_dim, images_out.size()[0], self.hidden_dim).requires_grad_()


        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch

        _, (hn, cn) = self.image_lstm(images_out, (images_lstm_hidden_state.detach(), images_lstm_cell_state.detach()))
        out, (hn, cn) = self.lstm(embeddings, (hn.detach(), cn.detach()))

        # Index hidden state of last time step
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])

        soft_max = self.soft_max(out)
        return soft_max
