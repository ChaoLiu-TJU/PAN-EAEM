import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale=230):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1 / math.sqrt(scale)

    def forward(self, q, k, v):
        # q, k, v: (B, L, D)
        attention = torch.matmul(q, k.transpose(-1, -2))
        attention = attention * self.scale
        attention = self.softmax(attention)
        context = torch.matmul(attention, v)
        return context, attention


class Attention(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_heads = 4
        self.d_k = self.hidden_size
        self.Q = nn.Linear(self.hidden_size, self.d_k * self.num_heads, bias=False)
        self.K = nn.Linear(self.hidden_size, self.d_k * self.num_heads, bias=False)
        self.V = nn.Linear(self.hidden_size, self.d_k * self.num_heads, bias=False)
        self.dot = ScaledDotProductAttention()
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.fc = nn.Linear(self.d_k * self.num_heads, self.hidden_size)

    def forward(self, support, query):
        B = support.size(0)
        N = support.size(1)
        K = support.size(2)

        q_support = support.view(B, -1, self.hidden_size)
        k_support = support.view(B, -1, self.hidden_size)
        v_support = support.view(B, -1, self.hidden_size)
        residual_support = q_support
        q_support = self.Q(q_support).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_support = self.K(k_support).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_support = self.V(v_support).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        support_, attention_query = self.dot(q_support, k_support, v_support)
        support_ = self.fc(support_)
        support_add = support_ + residual_support
        support_minus = residual_support-support_
        support_ = torch.cat((support_add, support_minus), -1)
        support_ = support_.view(B, N, K, -1)

        q_query = query.view(B, -1, self.hidden_size * 4)
        residual_query = q_query
        q_query = self.Q(q_query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        query_, attention = self.dot(q_query, k_support, v_support)
        query_ = self.fc(query_)
        query_add = query_ + residual_query
        query_minus = residual_query-query_
        query_ = torch.cat((query_add, query_minus), -1)
        return support_, query_


