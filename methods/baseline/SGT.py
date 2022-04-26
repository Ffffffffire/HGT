import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGTLayer(nn.Module):
    def __init__(self, embeddings_dimension, nheads=2, dropout=0.5):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''

        super(GTLayer, self).__init__()

        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension
        self.dropout = dropout

        self.head_dim = self.embeddings_dimension // self.nheads

        self.linear_k = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads)
        self.linear_v = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads)
        self.linear_q = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension)
        self.dropout = nn.Dropout(self.dropout)

        self.FFN1 = nn.Linear(embeddings_dimension, embeddings_dimension)
        self.FFN2 = nn.Linear(embeddings_dimension, embeddings_dimension)
        self.fdropout = nn.Dropout(p=dropout)
        self.LN1 = nn.LayerNorm(embeddings_dimension)
        self.LN2 = nn.LayerNorm(embeddings_dimension)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, h):
        q = self.linear_q(h)
        k = self.linear_k(h)
        v = self.linear_v(h)
        batch_size = k.size()[0]

        q_ = q.view(batch_size, self.nheads, -1, self.head_dim)
        k_ = k.view(batch_size, self.nheads, -1, self.head_dim)
        v_ = v.view(batch_size, self.nheads, -1, self.head_dim)

        batch_size, head, length, d_tensor = k_.size()
        k_t = k_.view(batch_size, head, d_tensor, length)
        score = (q_ @ k_t) / math.sqrt(d_tensor)
        score = F.softmax(score)
        context = score @ v_

        h_sa = context.view(batch_size, -1, self.head_dim * self.nheads)
        h_sa = self.linear_final(h_sa)
        h_sa = self.dropout(h_sa)

        h1 = self.LN1(h_sa + h)
        h1 = self.dropout1(h1)

        
        hf = self.FFN1(h1)
        hf = self.fdropout(F.relu(hf))
        hf = self.FFN2(hf)

        h2 = self.LN2(hf + h1)
        h2 = self.dropout2(h2)


        return h2


class SGT(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64, num_layers=8, nheads=2, dropout=0):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''

        super(SGT, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, embeddings_dimension, bias=True) for in_dim in input_dimensions])
        self.dropout = dropout

        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                SGTLayer(self.embeddings_dimension, self.nheads, self.dropout))
        self.Prediction = nn.Linear(embeddings_dimension, num_class)

    def forward(self, features_list, seqs, usemean = False):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h = h[seqs]
        for layer in range(self.num_layers):
            h = self.GTLayers[layer](h)
        if usemean:
            output = F.relu(self.Prediction(h)).mean(dim=1)
        else:
            output = F.relu(self.Prediction(h))[:,0,:]
        return output
