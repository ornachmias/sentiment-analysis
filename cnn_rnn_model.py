import torch
from torch import nn

# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
from torchvision.models import densenet121

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CombinedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(CombinedModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.image_lstm = nn.LSTM(input_size=1024, hidden_size=hidden_dim, batch_first=True)

        # get the pretrained densenet model
        self.densenet = densenet121(pretrained=True)

        # replace the classifier with a fully connected embedding layer
        self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)

        # add another fully connected layer
        self.embed = nn.Linear(in_features=1024, out_features=1024)

        # dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # activation layers
        self.prelu = nn.PReLU()

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, sentences, images):
        densenet_outputs = self.dropout(self.prelu(self.densenet(images)))

        # pass through the fully connected
        images_out = self.embed(densenet_outputs)

        embeddings = self.word_embeddings(sentences.to(torch.int64))

        images_lstm_hidden_state = torch.zeros((1, images.size()[0], self.hidden_dim)).requires_grad_().to(DEVICE)
        images_lstm_cell_state = torch.zeros((1, images_out.size()[0], self.hidden_dim)).requires_grad_().to(DEVICE)

        images_out = images_out.view(images_out.size()[0], 1, -1)
        out, (hn, cn) = self.image_lstm(images_out, (images_lstm_hidden_state.detach(), images_lstm_cell_state.detach()))
        out, (hn, cn) = self.lstm(embeddings, (hn.detach(), cn.detach()))

        # Index hidden state of last time step
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])

        soft_max = self.soft_max(out)
        return soft_max
