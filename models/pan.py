import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from fewshot_re_kit.att import Attention


class Sen_Att(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.d_k = hidden_size * 8
        self.Q = nn.Linear(self.hidden_size * 4, self.d_k, bias=False)
        self.K = nn.Linear(self.hidden_size * 4, self.d_k, bias=False)
        self.V = nn.Linear(self.hidden_size * 4, self.d_k, bias=False)
        self.layer_norm = nn.LayerNorm(self.hidden_size*4)
        self.fc = nn.Linear(self.hidden_size * 8, self.hidden_size*4)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q_x = self.Q(x)
        k_x = self.K(x)
        v_x = self.V(x)
        attention = torch.matmul(q_x, k_x.transpose(-1, -2))
        attention = self.softmax(attention)
        att_x = torch.matmul(attention, v_x)
        att_x = self.layer_norm(x + F.relu(self.fc(att_x)))
        return att_x


class PAN(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, entity_encoder, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder, entity_encoder)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size * 3, hidden_size)
        self.att = Sen_Att(hidden_size)
        self.hybrid = Attention(hidden_size)
        self.drop = nn.Dropout(0)

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support_word = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        support_head, support_tail = self.entity_encoder(support)  # (B * N * K, D), where D is the hidden size
        query_word = self.sentence_encoder(query)  # (B * N * Q, D)
        query_head, query_tail = self.entity_encoder(query)
        support = torch.cat((support_word, support_head, support_tail), 1)
        query = torch.cat((query_word, query_head, query_tail), 1)
        support = support.view(-1, N, K, self.hidden_size * 4)  # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size * 4)  # (B, N * Q, D)

        support, query = self.hybrid(support, query)

        query_res = query
        NQ = query.size(1)

        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1)
        query = query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1)
        alpha = torch.cat((query, support), -1)
        alpha = torch.relu(self.fc1(alpha))
        alpha = self.fc2(alpha).squeeze(-1)
        alpha = F.softmax(alpha, dim=-1)
        support_proto = (support * alpha.unsqueeze(4).expand(-1, -1, -1, -1, self.hidden_size * 4)).sum(3)
        logits = -self.__batch_dist__(support_proto, query_res)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred
