# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, copy, time


class CNNmodel(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_layer,
                 dropout,
                 use_use_cuda=True):
        super(CNNmodel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.use_cuda = use_use_cuda

        self.cnn_layer0 = nn.Conv1d(self.input_dim,
                                    self.hidden_dim,
                                    kernel_size=1,
                                    padding=0)
        self.cnn_layers = [
            nn.Conv1d(self.hidden_dim,
                      self.hidden_dim,
                      kernel_size=3,
                      padding=1) for i in range(self.num_layer - 1)
        ]
        self.drop = nn.Dropout(dropout)

        if self.use_cuda:
            self.cnn_layer0 = self.cnn_layer0.cuda()
            for i in range(self.num_layer - 1):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()

    def forward(self, input_feature):

        batch_size = input_feature.shape[0]
        seq_len = input_feature.shape[1]

        input_feature = input_feature.transpose(2, 1).contiguous()
        cnn_output = self.cnn_layer0(input_feature)
        cnn_output = self.drop(cnn_output)
        cnn_output = torch.tanh(cnn_output)

        for layer in range(self.num_layer - 1):
            cnn_output = self.cnn_layers[layer](cnn_output)
            cnn_output = self.drop(cnn_output)
            cnn_output = torch.tanh(cnn_output)

        cnn_output = cnn_output.transpose(2, 1).contiguous()
        return cnn_output


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,
                        value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query,
                                 key,
                                 value,
                                 mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class AttentionModel(nn.Module):

    def __init__(self, d_input, d_model, d_ff, head, num_layer, dropout):
        super(AttentionModel, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(head, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.layers = clones(layer, num_layer)
        self.norm = LayerNorm(layer.size)
        self.posi = PositionalEncoding(d_model, dropout)
        self.input2model = nn.Linear(d_input, d_model)

    def forward(self, x, mask):
        x = self.posi(self.input2model(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class NERmodel(nn.Module):

    def __init__(self,
                 model_type,
                 input_dim,
                 hidden_dim,
                 num_layer,
                 dropout=0.5,
                 use_cuda=True,
                 biflag=True):
        super(NERmodel, self).__init__()
        self.model_type = model_type

        if self.model_type == 'lstm':
            self.lstm = nn.LSTM(input_dim,
                                hidden_dim,
                                num_layers=num_layer,
                                batch_first=True,
                                bidirectional=biflag)
            self.drop = nn.Dropout(dropout)

        if self.model_type == 'cnn':
            self.cnn = CNNmodel(input_dim, hidden_dim, num_layer, dropout, use_cuda)

        if self.model_type == 'transformer':
            self.attention_model = AttentionModel(d_input=input_dim,
                                                  d_model=hidden_dim,
                                                  d_ff=2 * hidden_dim,
                                                  head=4,
                                                  num_layer=num_layer,
                                                  dropout=dropout)
            for p in self.attention_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, input, mask=None):

        if self.model_type == 'lstm':
            hidden = None
            feature_out, hidden = self.lstm(input, hidden)

            feature_out_d = self.drop(feature_out)

        if self.model_type == 'cnn':
            feature_out_d = self.cnn(input)

        if self.model_type == 'transformer':
            feature_out_d = self.attention_model(input, mask)

        return feature_out_d
