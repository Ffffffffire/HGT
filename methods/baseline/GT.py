import math

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv, GraphConv, SAGEConv


class GTLayer(nn.Module):
    def __init__(self, embeddings_dimension, ffn_dimension, nheads=2, dropout=0.5, activation = 'relu', rl = False, rl_dim=4):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''

        super(GTLayer, self).__init__()

        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension
        self.dropout = dropout

        self.head_dim = self.embeddings_dimension // self.nheads

        self.rl_dim = rl_dim

        self.linear_k = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias = False)
        self.linear_v = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias = False)
        self.linear_q = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias = False)

        if rl:
            self.r_k = nn.Linear(rl_dim, rl_dim, bias=False)
            self.r_q = nn.Linear(rl_dim, rl_dim, bias=False)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension, bias = False)
        self.dropout = nn.Dropout(self.dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU(0.2)
        
        self.FFN1 = nn.Linear(embeddings_dimension, ffn_dimension)
        self.FFN2 = nn.Linear(ffn_dimension, embeddings_dimension)
        self.dropout = nn.Dropout(p=dropout)
        self.LN1 = nn.LayerNorm(embeddings_dimension)
        self.LN2 = nn.LayerNorm(embeddings_dimension)
        #self.BN1 = nn.BatchNorm1d(100)
        #self.BN2 = nn.BatchNorm1d(100)

        #self.reset_parameters(bias=True)

    def reset_parameters(self, bias = False):
        #gain = nn.init.calculate_gain('relu')
        #nn.init.kaiming_uniform_(self.linear_q.weight, nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.linear_k.weight, nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.linear_v.weight, nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.linear_final.weight, nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.FFN1.weight, nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.FFN2.weight, nonlinearity='relu')
        if bias == True:
            #nn.init.constant_(self.linear_q.bias, 0)
            #nn.init.constant_(self.linear_k.bias, 0)
            #nn.init.constant_(self.linear_v.bias, 0)
            nn.init.constant_(self.linear_final.bias, 0)
            nn.init.constant_(self.FFN1.bias, 0)
            nn.init.constant_(self.FFN2.bias, 0)

    def forward(self, h, rh=None, mask=None, e=1e-12):
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
        if rh is not None:
            r_k = self.r_k(rh)
            r_q = self.r_q(rh)
            r_k_ = r_k.unsqueeze(1)
            r_q_ = r_q.unsqueeze(1)
            r_k_t = r_k_.view(batch_size, 1, self.rl_dim, length)
            score += (r_q_ @ r_k_t) / 2

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        score = F.softmax(score, dim = -1)
        context = score @ v_

        h_sa = context.view(batch_size, -1, self.head_dim * self.nheads)
        h_sa = self.linear_final(h_sa)
        h_sa = self.dropout(h_sa)

        h1 = self.LN1(h_sa + h)
        h1 = self.dropout(h1)
        
        hf = self.activation(self.FFN1(h1))
        hf = self.dropout(hf)
        hf = self.FFN2(hf)

        h2 = self.LN2(h1+hf)
        h2 = self.dropout(h2)

        return h2

class GT(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64, ffn_dimension = 128, num_layers=8, nheads=2, dropout=0, activation = 'relu', num_glo = 0):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''

        super(GT, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(
            in_dim, embeddings_dimension, bias=False) for in_dim in input_dimensions])
        self.glo = num_glo > 0

        if self.glo:     
            self.globalembedding = torch.nn.Parameter(torch.empty(num_glo, embeddings_dimension))
            nn.init.xavier_normal_(self.globalembedding)

        self.dropout = dropout

        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension, ffn_dimension, self.nheads, self.dropout, activation=activation))
        self.Prediction = nn.Linear(embeddings_dimension, num_class, bias = False)
        #self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

        for fc in self.fc_list:
            nn.init.kaiming_uniform_(fc.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.Prediction.weight, nonlinearity='relu')

    def forward(self, features_list, seqs, norm=False):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h = h[seqs]
        if self.glo:
            h = torch.cat([h,self.globalembedding.expand(h.shape[0],-1,-1)],dim=1)
        for layer in range(self.num_layers):
            h = self.GTLayers[layer](h)
        #h = h[:,0,:] + h[:,1:,:].mean(dim=1)
        output = self.Prediction(h[:,0,:])
        if norm:
            output = output / (torch.norm(output, dim=1, keepdim=True)+1e-12)
        return output

class GT_SSL(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64, ffn_dimension=128, num_layers=8, nheads=2, dropout=0, activation='relu', num_glo=0):
        '''
            embeddings_dimension: d = dp = dk = dq
            multi-heads: n
            
        '''

        super(GT_SSL, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(
            in_dim, embeddings_dimension, bias=False) for in_dim in input_dimensions])
        self.glo = num_glo > 0

        if self.glo:
            self.globalembedding = torch.nn.Parameter(
                torch.empty(num_glo, embeddings_dimension))
            nn.init.xavier_normal_(self.globalembedding)

        self.dropout = dropout

        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension, ffn_dimension, self.nheads, self.dropout, activation=activation))
        self.Prediction = nn.Linear(
            embeddings_dimension, num_class, bias=False)
        self.ssl_pre = nn.Sequential(nn.Linear(embeddings_dimension, embeddings_dimension, bias=False), nn.ReLU(), 
            nn.Linear(embeddings_dimension, 1), bias=False)
        #self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

        for fc in self.fc_list:
            nn.init.kaiming_uniform_(fc.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.Prediction.weight, nonlinearity='relu')

    def forward(self, features_list, seqs, norm=False):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h = h[seqs]
        if self.glo:
            h = torch.cat(
                [h, self.globalembedding.expand(h.shape[0], -1, -1)], dim=1)
        for layer in range(self.num_layers):
            h = self.GTLayers[layer](h)
        output_ssl = self.ssl_pre(h).flatten()
        output = self.Prediction(h[:, 0, :])
        if norm:
            output = output / (torch.norm(output, dim=1, keepdim=True)+1e-12)
        return output, output_ssl

class RGT(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64, ffn_dimension=128, num_layers=8, nheads=4, dropout=0.5, rl_dimension=4, ifcat=True, GNN='SAGE', activation='relu', num_glo = 0):

        super(RGT, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.ifcat = ifcat
        self.gnn = GNN

        if self.ifcat:
            self.fc_list = nn.ModuleList([nn.Linear(
                in_dim, embeddings_dimension-rl_dimension, bias=False) for in_dim in input_dimensions])
        else:
            self.fc_list = nn.ModuleList([nn.Linear(
                in_dim, embeddings_dimension, bias=False) for in_dim in input_dimensions])
        self.dropout = dropout
        self.glo = num_glo > 0

        if self.glo:
            gain = nn.init.calculate_gain('relu')     
            self.globalembedding = torch.nn.Parameter(torch.empty(num_glo, embeddings_dimension))
            nn.init.xavier_normal_(self.globalembedding, gain)

        #self.rl_first = nn.Linear(4, rl_dimension, bias=False)

        if self.gnn == 'SAGE':
            self.rl_self = nn.Linear(rl_dimension, rl_dimension // 2, bias=False)
            self.rl_neighbor = nn.Linear(rl_dimension, rl_dimension // 2, bias=False)
        elif self.gnn == 'GCN':
            self.rl_w = nn.Linear(rl_dimension, rl_dimension, bias=False)
        elif self.gnn == 'GIN':
            self.rl_w = nn.Sequential(
                nn.Linear(rl_dimension, rl_dimension, bias=False), nn.ReLU, nn.Linear(rl_dimension, rl_dimension, bias=False))

        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension, ffn_dimension, self.nheads, self.dropout, activation, self.ifcat, rl_dimension))
        
        self.Prediction = nn.Linear(embeddings_dimension, num_class, bias = False)
        

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        for fc in self.fc_list:
            reset(fc.weight)

        reset(self.rl_self.weight)
        reset(self.rl_neigh.weight)
        reset(self.Prediction.weight)

    def forward(self, features_list, seqs, type_emb, node_type, adjs, K=3, norm=False):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        r = self.relational_encoding(seqs, type_emb, node_type, adjs, K, norm)
        h = h[seqs]
        if self.ifcat:
            h = torch.cat([h,r],dim=2)
        #h = h[seqs]
        if self.glo:
            h = torch.cat(
                [h, self.globalembedding.expand(h.shape[0], -1, -1)], dim=1)
        for layer in range(self.num_layers):
            if self.ifcat:
                h = self.GTLayers[layer](h, r)
            else:
                h = self.GTLayers[layer](h, r)
        output = self.Prediction(h[:, 0, :])
        if norm:
            output = output / (torch.norm(output, dim=1, keepdim=True)+1e-12)
        return output

    def relational_encoding(self, seqs, type_emb, node_type, adjs, K, norm = False):
        node_type = [i for i, z in zip(
            range(len(node_type)), node_type) for x in range(z)]
        r_0 = type_emb[node_type]
        r = r_0[seqs]
        #r = self.rl_first(r_0)       
        masks = torch.zeros(adjs.shape).to(torch.device('cuda:0'))
        r[:, 0, :] = r[:, 0, :] * 2
        for _ in range(K):
            source_id = torch.nonzero(r.sum(dim=2) != 1)
            masks[source_id[0], :, source_id[1]] = 1
            madjs = torch.mul(adjs, masks)
            neighbor = torch.matmul(madjs, r)
            output_self = self.rl_self(r)
            output_neighbor = self.rl_neighbor(neighbor)
            r = torch.cat([output_self, output_neighbor], dim=2)
            #r = F.relu(r)
            if norm:
                r =  r / (torch.norm(r, dim=2, keepdim=True)+1e-12)
        return r


'''
class RGT(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64, ffn_dimension=128, num_layers=8, nheads=2, dropout=0, rl_type='GCN', rl_node = False, is_add = False):

        super(RGT, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        if is_add:
            self.fc_list = nn.ModuleList([nn.Linear(
                in_dim, embeddings_dimension, bias=False) for in_dim in input_dimensions])
        else:
            self.fc_list = nn.ModuleList([nn.Linear(in_dim, embeddings_dimension-4, bias=False) for in_dim in input_dimensions])
        self.dropout = dropout
        self.rl_node = rl_node
        self.is_add = is_add

        if rl_type == 'GCN':
            self.rl_layer = GraphConv(4, 4, activation=F.elu, weight=True)
        elif rl_type == 'SAGE':
            self.rl_layer = SAGEConv(4, 4, 'mean', activation=F.elu, norm=F.normalize)
        else:
            mlp = nn.Sequential(nn.Linear(4, 4, bias=False),nn.ReLU(),nn.Linear(4, 4, bias=False))
            self.rl_layer = GINConv(mlp, 'mean')

        self.rl_self = nn.Linear(4, 4, bias=False)
        self.rl_neigh = nn.Linear(4, 4, bias=False)

        self.add_layer = nn.Linear(4, embeddings_dimension, bias=False)

        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension, ffn_dimension, self.nheads, self.dropout))
        self.Prediction = nn.Linear(embeddings_dimension, num_class, bias=False)
    
    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        for fc in self.fc_list:
            reset(fc.weight)

        reset(self.rl_self.weight)
        reset(self.rl_neigh.weight)
        reset(self.Prediction.weight)

    def forward(self, graph, features_list, seqs, type_emb, node_type, usemean = False, K = 3):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        if self.rl_node:
            r = self.relational_encoding_node(graph, type_emb, node_type, K)
        else:
            r = self.relational_encoding(graph, type_emb, node_type, K)
        if self.is_add:
            h = h + self.add_layer(r)
        else:
            h = torch.cat([h,r],1)
        h = h[seqs]
        for layer in range(self.num_layers):
            h = self.GTLayers[layer](h)
        if usemean:
            output = F.relu(self.Prediction(h)).mean(dim=1)
        else:
            output = F.relu(self.Prediction(h))[:,0,:]
        return output
    

    def relational_encoding(self, graph, type_emb, node_type, K):
        node_type = [i for i,z in zip(range(len(node_type)), node_type) for x in range(z)]
        r_0 = type_emb[node_type]
        r = r_0
        for _ in range(K):
            r = self.rl_layer(graph,r)
        #print(torch.cuda.memory_allocated())
        return r
    
    def relational_encoding_node(self, graph, type_emb, node_type, K):
        node_type = [i for i, z in zip(
            range(len(node_type)), node_type) for x in range(z)]
        r_0 = type_emb[node_type]
        r = []
        cnt = 0
        for x in range(graph.num_nodes()):
            r_x = r_0
            r_x[x] += 1
            with graph.local_scope():
                for _ in range(K):
                    n_list = torch.nonzero(r_x.sum(dim=1) != 1).squeeze()
                    graph.ndata['r'] = r_x
                    graph.push(n_list, fn.copy_u('r', 'm'), fn.sum('m', 'neigh'))
                    r_n = graph.ndata['neigh']
                    r_x = F.relu(self.rl_self(r_x) + self.rl_neigh(r_n))
            r.append(r_x.mean(dim=0))
            #print(torch.cuda.memory_allocated())
        r = torch.cat(r, 0)
        return r


class RGT_v2(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64, ffn_dimension=128, num_layers=8, nheads=2, dropout=0, rl_type='GCN', rl_node=False, is_add=False):

        super(RGT_v2, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(
            in_dim, embeddings_dimension, bias=False) for in_dim in input_dimensions])
        self.dropout = dropout
        self.rl_node = rl_node
        self.is_add = is_add

        if rl_type == 'GCN':
            self.rl_layer = GraphConv(4, 4, activation=F.elu, weight=True)
        elif rl_type == 'SAGE':
            self.rl_layer = SAGEConv(
                4, 4, 'mean', activation=F.elu, norm=F.normalize)
        else:
            mlp = nn.Sequential(nn.Linear(4, 4, bias=False),
                                nn.ReLU(), nn.Linear(4, 4, bias=False))
            self.rl_layer = GINConv(mlp, 'mean')

        self.rl_self = nn.Linear(4, 4, bias=False)
        self.rl_neigh = nn.Linear(4, 4, bias=False)

        self.add_layer = nn.Linear(4, embeddings_dimension, bias=False)

        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension, ffn_dimension, self.nheads, self.dropout))
        self.Prediction = nn.Linear(
            embeddings_dimension, num_class, bias=False)

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        for fc in self.fc_list:
            reset(fc.weight)

        reset(self.rl_self.weight)
        reset(self.rl_neigh.weight)
        reset(self.Prediction.weight)

    def forward(self, graph, features_list, seqs, type_emb, node_type, usemean=False, K=3):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        if self.rl_node:
            r = self.relational_encoding_node(graph, seqs, type_emb, node_type, K)
        else:
            r = self.relational_encoding(graph, type_emb, node_type, K)
        h = h[seqs]
        if not self.rl_node:
            r = r[seqs]
        for layer in range(self.num_layers):
            h = self.GTLayers[layer].rl_attention(h,r)
        if usemean:
            output = F.relu(self.Prediction(h)).mean(dim=1)
        else:
            output = F.relu(self.Prediction(h))[:, 0, :]
        return output

    def relational_encoding(self, graph, type_emb, node_type, K):
        node_type = [i for i, z in zip(
            range(len(node_type)), node_type) for x in range(z)]
        r_0 = type_emb[node_type]
        r = r_0
        for _ in range(K):
            r = self.rl_layer(graph, r)
        #print(torch.cuda.memory_allocated())
        return r

    def relational_encoding_node(self, graph, seqs, type_emb, node_type, K):
        node_type = [i for i, z in zip(
            range(len(node_type)), node_type) for x in range(z)]
        r_0 = type_emb[node_type]
        r = []
        length = seqs.shape[0]
        for i in range(length):
            r_x = r_0[seqs[i]]
            r_x[0] += 1
            #print(graph.device)
            sg = dgl.node_subgraph(graph, seqs[i])
            for _ in range(K):
                n_list = torch.nonzero(r_x.sum(dim=1) != 1).squeeze()
                sg.ndata['r'] = r_x
                sg.push(n_list, fn.copy_u(
                    'r', 'm'), fn.sum('m', 'neigh'))
                r_n = sg.ndata['neigh']
                r_x = F.relu(self.rl_self(r_x) + self.rl_neigh(r_n))
            r.append(r_x)
        r = torch.cat(r, 0).reshape(length, -1, 4)
        return r


class RGT_v3(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64, ffn_dimension=128, num_layers=8, nheads=2, dropout=0, rl_dimension = 4):

        super(RGT_v3, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(
            in_dim, embeddings_dimension, bias=False) for in_dim in input_dimensions])
        self.dropout = dropout

        self.rl_first = nn.Linear(4, rl_dimension)

        self.rl_self = nn.Linear(rl_dimension, rl_dimension // 2) 
        self.rl_neighbor = nn.Linear(rl_dimension, rl_dimension // 2)

        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension, ffn_dimension, self.nheads, self.dropout, rl_dimension))
        self.Prediction = nn.Linear(
            embeddings_dimension, num_class, bias=False)

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        for fc in self.fc_list:
            reset(fc.weight)

        reset(self.rl_self.weight)
        reset(self.rl_neigh.weight)
        reset(self.Prediction.weight)

    def forward(self, features_list, seqs, type_emb, node_type, adjs, K=3):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        r = self.relational_encoding(seqs, type_emb, node_type, adjs, K)
        h = h[seqs]
        for layer in range(self.num_layers):
            h = self.GTLayers[layer].rl_attention(h, r)
        output = F.relu(self.Prediction(h))[:, 0, :]
        return output

    def relational_encoding(self, seqs, type_emb, node_type, adjs, K):
        node_type = [i for i, z in zip(
            range(len(node_type)), node_type) for x in range(z)]
        r_0 = type_emb[node_type]
        r_0 = r_0[seqs]
        r = r_0
        for _ in range(K):
            neighbor = torch.matmul(adjs, r)
            output_self = self.rl_self(r)
            output_neighbor = self.rl_neighbor(neighbor)
            r = torch.cat([output_self, output_neighbor], dim=2)
            #r = F.normalize(F.relu(r))
            r = F.relu(r)
        return r


class RGT_v4(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64, ffn_dimension=128, num_layers=8, nheads=2, dropout=0, rl_dimension=4):

        super(RGT_v4, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(
            in_dim, embeddings_dimension, bias=False) for in_dim in input_dimensions])
        self.dropout = dropout

        self.rl_first = nn.Linear(4, rl_dimension)

        self.rl_self = nn.Linear(rl_dimension, rl_dimension // 2, bias=False)
        self.rl_neighbor = nn.Linear(rl_dimension, rl_dimension // 2, bias=False)

        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension, ffn_dimension, self.nheads, self.dropout, rl_dimension))
        self.Prediction = nn.Linear(embeddings_dimension, num_class, bias=False)
        #self.Prediction = nn.Sequential(nn.Linear(embeddings_dimension, embeddings_dimension, bias=False),nn.ReLU(), nn.Linear(embeddings_dimension, num_class, bias=False))


    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        for fc in self.fc_list:
            reset(fc.weight)

        reset(self.rl_self.weight)
        reset(self.rl_neigh.weight)
        reset(self.Prediction.weight)

    def forward(self, features_list, seqs, type_emb, node_type, adjs, K=3):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        r = self.relational_encoding(seqs, type_emb, node_type, adjs, K)
        h = h[seqs]
        #output = []
        for layer in range(self.num_layers):
            h = self.GTLayers[layer].rl_attention(h, r)
            #output.append(h[:,0,:])
        #output = torch.cat(output,dim=-1).
        #h = h[:,0,:] + h[:,1:,:].mean(dim=1)
        output = F.relu(self.Prediction(h))[:,0,:]
        return output

    def relational_encoding(self, seqs, type_emb, node_type, adjs, K):
        node_type = [i for i, z in zip(
            range(len(node_type)), node_type) for x in range(z)]
        r_0 = type_emb[node_type]
        r_0 = r_0[seqs]
        r = r_0
        masks = torch.zeros(adjs.shape).to(torch.device('cuda:0'))
        r[:, 0, :] = r[:, 0, :] * 2
        for _ in range(K):
            source_id = torch.nonzero(r.sum(dim=2) != 1)
            masks[source_id[0], :, source_id[1]] = 1
            madjs = torch.mul(adjs, masks)
            neighbor = torch.matmul(madjs, r)
            output_self = self.rl_self(r)
            output_neighbor = self.rl_neighbor(neighbor)
            r = torch.cat([output_self, output_neighbor], dim=2)
            r = F.relu(r)
        return r


class RGT_v5(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64, ffn_dimension=128, num_layers=8, nheads=2, dropout=0, rl_dimension=4):

        super(RGT_v5, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(
            in_dim, embeddings_dimension, bias=False) for in_dim in input_dimensions])
        self.dropout = dropout

        self.rl_first = nn.Linear(4, rl_dimension)

        self.rl_self = nn.Linear(rl_dimension, rl_dimension // 2)
        self.rl_neighbor = nn.Linear(rl_dimension, rl_dimension // 2)

        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                GTLayer(self.embeddings_dimension, ffn_dimension, self.nheads, self.dropout, rl_dimension))
        self.Prediction = nn.Linear(
            embeddings_dimension, num_class, bias=False)

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        for fc in self.fc_list:
            reset(fc.weight)

        reset(self.rl_self.weight)
        reset(self.rl_neigh.weight)
        reset(self.Prediction.weight)

    def forward(self, features_list, node_idx, seqs, type_emb, node_type, adjs, K=3):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        r = self.relational_encoding(seqs, type_emb, node_type, adjs, K)
        h = h[seqs[node_idx]]
        r = r[seqs[node_idx]]
        for layer in range(self.num_layers):
            h = self.GTLayers[layer].rl_attention(h, r)
        output = F.relu(self.Prediction(h))[:, 0, :]
        return output

    def relational_encoding(self, seqs, type_emb, node_type, adjs, K):
        node_type = [i for i, z in zip(
            range(len(node_type)), node_type) for x in range(z)]
        r_0 = type_emb[node_type]
        r_0 = r_0[seqs]
        r = r_0
        masks = torch.zeros(adjs.shape).to(torch.device('cuda:0'))
        r[:, 0, :] = r[:, 0, :] * 2
        for _ in range(K):
            source_id = torch.nonzero(r.sum(dim=2) != 1)
            masks[source_id[0], :, source_id[1]] = 1
            madjs = torch.mul(adjs, masks)
            neighbor = torch.matmul(madjs, r)
            output_self = self.rl_self(r)
            output_neighbor = self.rl_neighbor(neighbor)
            r = torch.cat([output_self, output_neighbor], dim=2)
            r = F.relu(r)
        r = r[:,0,:]
        return r

class AGTLayer(nn.Module):
    def __init__(self, embeddings_dimension, ffn_dimension, nheads=2, dropout=0.5, rl=False, rl_dim=4):

        super(AGTLayer, self).__init__()

        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension
        self.dropout = dropout

        self.head_dim = self.embeddings_dimension // self.nheads

        self.rl_dim = rl_dim

        self.linear_w = nn.Linear(self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.att_source = nn.Linear(self.head_dim * self.nheads, self.nheads, bias = False)
        self.att_target = nn.Linear(self.head_dim * self.nheads, self.nheads, bias = False)

        if rl:
            self.r_k = nn.Linear(rl_dim, rl_dim, bias=False)
            self.r_q = nn.Linear(rl_dim, rl_dim, bias=False)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension, bias=False)
        self.dropout = nn.Dropout(self.dropout)

        self.FFN1 = nn.Linear(embeddings_dimension, ffn_dimension, bias=True)
        self.FFN2 = nn.Linear(ffn_dimension, embeddings_dimension, bias=True)
        self.fdropout = nn.Dropout(p=dropout)
        self.LN1 = nn.LayerNorm(embeddings_dimension)
        self.LN2 = nn.LayerNorm(embeddings_dimension)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        #self.reset_parameters()

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.linear_k.weight)
        reset(self.linear_q.weight)
        reset(self.linear_v.weight)
        reset(self.linear_final.weight)
        reset(self.FFN1.weight)
        reset(self.FFN2.weight)

    def forward(self, h, rh=None, mask=None, e=1e-12):
        fh = self.linear_w(h)
        batch_size = fh.size()[0]
        fh_ = fh.view(batch_size, self.nheads, -1, self.head_dim)
        

        a_source = self.att_source(fh).view(batch_size, self.nheads, -1, 1)
        a_target = self.att_target(fh).view(batch_size, self.nheads, 1, -1)
        
        score = F.softmax(a_source + a_target, dim=-1)

        if rh is not None:
            r_k = self.r_k(rh)
            r_q = self.r_q(rh)
            r_k_ = r_k.unsqueeze(1)
            r_q_ = r_q.unsqueeze(1)
            r_k_t = r_k_.view(batch_size, 1, self.rl_dim, length)
            score += (r_q_ @ r_k_t) / 2

        context = score @ fh_

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

class AGT(nn.Module):
    def __init__(self, num_class, input_dimensions, embeddings_dimension=64, ffn_dimension=128, num_layers=8, nheads=2, dropout=0, pre=0, pro=0, num_glo=0):

        super(AGT, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(
            in_dim, embeddings_dimension, bias=False) for in_dim in input_dimensions])
        self.glo = num_glo > 0
        self.pre = pre
        self.pro = pro

        if self.glo:
            gain = nn.init.calculate_gain('relu')
            self.globalembedding = torch.nn.Parameter(
                torch.empty(num_glo, embeddings_dimension))
            nn.init.xavier_normal_(self.globalembedding, gain)

        self.dropout = dropout

        if self.pre == 1:
            self.prelayers = nn.Linear(
                embeddings_dimension, embeddings_dimension, bias=False)
        elif self.pre == 2:
            self.prelayers = nn.Sequential(nn.Linear(embeddings_dimension, embeddings_dimension, bias=False), nn.ReLU(
            ), nn.Linear(embeddings_dimension, embeddings_dimension, bias=False))

        self.GTLayers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.GTLayers.append(
                AGTLayer(self.embeddings_dimension, ffn_dimension, self.nheads, self.dropout))
        if self.pro == 0:
            self.Prediction = nn.Linear(
                embeddings_dimension, num_class, bias=False)
        else:
            self.Prediction = nn.Sequential(nn.Linear(embeddings_dimension, embeddings_dimension, bias=False), nn.ReLU(
            ), nn.Linear(embeddings_dimension, num_class, bias=False))

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        for fc in self.fc_list:
            reset(fc.weight)

        reset(self.rl_self.weight)
        reset(self.rl_neigh.weight)
        reset(self.Prediction.weight)

    def forward(self, features_list, seqs, usemean=False, norm=False):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h = h[seqs]
        if self.glo:
            h = torch.cat(
                [h, self.globalembedding.expand(h.shape[0], -1, -1)], dim=1)
        for layer in range(self.num_layers):
            h = self.GTLayers[layer](h)
        output = self.Prediction(h[:, 0, :])
        if norm:
            output = output / (torch.norm(output, dim=1, keepdim=True)+1e-12)
        return output
'''
