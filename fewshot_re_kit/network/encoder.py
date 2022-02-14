import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim


class Encoder(nn.Module):
    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.conv = nn.Conv2d(1, self.hidden_size, kernel_size=(3, self.embedding_dim), padding=(1, 0))

    def forward(self, inputs):
        return self.cnn(inputs)

    def cnn(self, inputs):
        x = self.conv(inputs.transpose(1, 2))
        return x  # n x hidden_size


class Entity_Encoder(nn.Module):
    def __init__(self, entity_length, word_embedding_dim=50, hidden_size=230):
        nn.Module.__init__(self)

        self.entity_length = entity_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim
        self.conv = nn.Conv2d(1, self.hidden_size, kernel_size=(3, self.embedding_dim), padding=(1, 0))
        self.pool = nn.MaxPool1d(entity_length)

    def forward(self, head_emb, tail_emb):
        return self.cnn(head_emb, tail_emb)

    def cnn(self, head_emb, tail_emb):
        head_enc = self.conv(head_emb.unsqueeze(1)).squeeze(-1)
        # head_enc = F.relu(head_enc)
        head_enc = self.pool(head_enc)

        tail_enc = self.conv(tail_emb.unsqueeze(1)).squeeze(-1)
        # tail_enc = F.relu(tail_enc)
        tail_enc = self.pool(tail_enc)
        return head_enc.squeeze(2), tail_enc.squeeze(2)  # n x hidden_size
