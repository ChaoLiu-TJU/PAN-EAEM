import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        
        # Word embedding
        unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        blk = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0] + 2, self.word_embedding_dim, padding_idx=word_vec_mat.shape[0] + 1)
        self.word_embedding.weight.data.copy_(torch.cat((word_vec_mat, unk, blk), 0))

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

    def forward(self, inputs):
        word = inputs['word']
        pos1 = inputs['pos1']
        pos2 = inputs['pos2']
        
        x = torch.cat([self.word_embedding(word), 
                            self.pos1_embedding(pos1), 
                            self.pos2_embedding(pos2)], 2)
        return x


class Entity_Embedding(nn.Module):

    def __init__(self, word_vec_mat, entity_length, word_embedding_dim=50):
        nn.Module.__init__(self)

        self.max_length = entity_length
        self.word_embedding_dim = word_embedding_dim

        # Word embedding
        unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        blk = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0] + 2, self.word_embedding_dim,
                                           padding_idx=word_vec_mat.shape[0] + 1)
        self.word_embedding.weight.data.copy_(torch.cat((word_vec_mat, unk, blk), 0))

    def forward(self, inputs):
        head = inputs['head']
        tail = inputs['tail']

        head_emb = self.word_embedding(head)
        tail_emb = self.word_embedding(tail)

        return head_emb, tail_emb


