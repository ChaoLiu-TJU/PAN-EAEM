import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim
from . import network


class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, hidden_size)
        self.pool = nn.MaxPool1d(max_length)
        self.fc = nn.Linear(word_embedding_dim + pos_embedding_dim * 2, self.hidden_size)

    def forward(self, inputs):
        input_mask = (inputs['mask'] != 0).float()
        max_length = input_mask.long().sum(1).max().item()
        input_mask = input_mask[:, :max_length].contiguous()
        x_emb = self.embedding(inputs)
        x_emb = x_emb[:, :max_length].contiguous()
        x = self.fc(x_emb)
        x_enc = self.encoder(x_emb.unsqueeze(2)).squeeze(3)
        x_enc = (x_enc * input_mask.unsqueeze(1)).transpose(1, 2)
        x = torch.cat((x_enc, x), -1)
        x, _ = torch.max(x, 1)
        return x


class PCNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, hidden_size)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder.pcnn(x, inputs['mask'])
        return x


class CNNEntityEncoder(nn.Module):

    def __init__(self, word_vec_mat, entity_length, word_embedding_dim=50, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = entity_length
        self.embedding = network.embedding.Entity_Embedding(word_vec_mat, entity_length, word_embedding_dim)
        self.encoder = network.encoder.Entity_Encoder(entity_length, word_embedding_dim, hidden_size)

    def forward(self, inputs):
        head_emb, tail_emb = self.embedding(inputs)
        head_enc, tail_enc = self.encoder(head_emb, tail_emb)
        return head_enc, tail_enc
