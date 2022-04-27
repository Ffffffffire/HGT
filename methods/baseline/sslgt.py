import argparse
import os
import random
import sys
import time
from platform import node

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from GT import GT_SSL
from utils.data import load_data
from utils.pytorchtools import EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sys.path.append('../../')


#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def node2seq(graph, nodes, k):
    seqs = []
    for node in nodes:
        seq = set()  
        current_level = [node]
        seq.add(node)
        for _ in range(k):
            next_level = []
            for root in current_level:
                neighbor_list = graph.successors(root).numpy().tolist()
                next_level.extend(neighbor_list)
                seq.update(neighbor_list)
            current_level = next_level
        seqs.append(list(seq))
    return seqs

def node2seq_fixlength(graph, nodes, k, maxlen):
    seqs = []
    for node in nodes:
        seq = set()  
        current_level = [node]
        seq.add(node)
        for _ in range(k):
            next_level = []
            for root in current_level:
                neighbor_list = graph.successors(root).numpy().tolist()
                next_level.extend(neighbor_list)
                seq.update(neighbor_list)
            current_level = next_level
        seq = list(seq)[:maxlen]
        seqs.append(list(seq))
    return seqs

def len_seq_min_max(seqs):
    len_seq = torch.zeros(len(seqs))
    for i in range(len(seqs)):
        len_seq[i] = len(seqs[i])
    print("Max Len: %d Mean Len: %.4f Min Len: %d" % (len_seq.max().item(), len_seq.mean().item(), len_seq.min().item()))




def run_model_DBLP(args):
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(
        args.dataset, args.seed)
    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)

    all_nodes = np.arange(features_list[0].shape[0])
    
    node_seq = torch.zeros(features_list[0].shape[0], args.len_seq).long()
    dist_seq = torch.zeros(features_list[0].shape[0], args.len_seq)

    degree = g.in_degrees()

    n = 0

    for x in all_nodes:

        cnt = 0
        scnt = 0
        node_seq[n, cnt] = x
        cnt += 1
        start = node_seq[n, scnt].item()
        while (cnt < args.len_seq):
            sample_list = g.successors(start).numpy().tolist()
            nsampled = max(int(len(sample_list) * args.drop_ratio), 1)
            sampled_list = random.sample(sample_list, nsampled)
            for i in range(nsampled):
                node_seq[n, cnt] = sampled_list[i]
                dist_seq[n, cnt] = dist_seq[n, scnt] + 1
                cnt += 1
                if cnt == args.len_seq:
                    break
            scnt += 1
            start = node_seq[n, scnt].item()    
        n += 1

    train_seq = node_seq[train_idx]
    val_seq = node_seq[val_idx]
    test_seq = node_seq[test_idx]

    dist_seq = dist_seq[train_idx]
    degree_seq = degree[train_seq]
    node_type = [features.shape[0] for features in features_list]
    node_type = torch.tensor([i for i, z in zip(range(len(node_type)), node_type) for x in range(z)]).float()
    type_seq = node_type[train_seq]

    if args.ssl == 'dist':
        ssl_label = dist_seq.flatten().to(device)
    elif args.ssl == 'degree':
        ssl_label = degree_seq.flatten().to(device)
    elif args.ssl == 'type':
        ssl_label = type_seq.flatten().to(device)

    micro_f1 = torch.zeros(args.repeat)
    macro_f1 = torch.zeros(args.repeat)

    num_classes = dl.labels_train['num_classes']

    g = g.to(device)

    for i in range(args.repeat):
        
        net = GT_SSL(num_classes, in_dims, args.hidden_dim, args.ffn_dim, args.num_layers, args.num_heads, args.dropout,activation = 'leakyrelu', num_glo= args.num_g)

        net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                       save_path='checkpoint/gt_{}_{}.pt'.format(args.dataset, args.num_layers))
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()

            ssl_criterion = torch.nn.MSELoss()

            logits, ssl_output = net(features_list, train_seq, args.usenorm)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp, labels[train_idx])
            ssl_loss = ssl_criterion(ssl_output, ssl_label)
            loss = train_loss + args.lbd * ssl_loss

            # autograd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | supervise_Loss: {:.4f} | ssl_Loss: {:.4f} | Time: {:.4f}'.format(
                epoch, loss.item(), train_loss.item(), ssl_loss.item(), t_end-t_start))

            t_start = time.time()

            #print('Train torch.cuda.memory_allocated():',torch.cuda.memory_allocated() / 1024 / 1024)
            # validation
            net.eval()
            with torch.no_grad():
                #logits = net(features_list, val_seq, type_emb,node_type, val_adjs, args.K)
                logits,_ = net(features_list, val_seq, args.usenorm)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp, labels[val_idx])
                pred = logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)
                pred = onehot[pred]
                print(dl.evaluate_valid(pred, dl.labels_train['data'][val_idx]))

                #print('Valid torch.cuda.memory_allocated():',torch.cuda.memory_allocated() / 1024 / 1024)
    
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load(
            'checkpoint/gt_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        with torch.no_grad():
            logits,_ = net(features_list, test_seq, args.usenorm)
            test_logits = logits
            if args.mode == 1:
                pred = test_logits.cpu().numpy().argmax(axis=1)
                dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{i+1}.txt")
            else:
                pred = test_logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)
                pred = onehot[pred]
                result = dl.evaluate_valid(pred, dl.labels_test['data'][test_idx])
                print(result)
                micro_f1[i] = result['micro-f1']
                macro_f1[i] = result['macro-f1']
    print('Micro-f1: %.4f, std: %.4f' % (micro_f1.mean().item(), micro_f1.std().item()))
    print('Macro-f1: %.4f, std: %.4f' % (macro_f1.mean().item(), macro_f1.std().item()))
    #dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{i+1}.txt")
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                    '4 - only term features (id vec for others);' +
                    '5 - only term features (zero vec for others).')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=32,
                    help='Dimension of the node hidden state. Default is 32.')
    ap.add_argument('--ffn-dim', type=int, default=64,
                    help='Dimension of the FFN. Default is 64.')
    ap.add_argument('--num-g', type=int, default=0)
    ap.add_argument('--rl-dim', type=int, default=4,
                    help='Dimension of the rl layer. Default is 4.')
    ap.add_argument('--num-heads', type=int, default=8,
                    help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=100, help='Patience.')
    ap.add_argument('--repeat', type=int, default=5,
                    help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=0)
    ap.add_argument('--len-seq', type=int, default=120)
    ap.add_argument('--drop-ratio', type=float, default=1)
    ap.add_argument('--usenorm', type=bool, default=False)
    ap.add_argument('--rl_type', type=str, default='GIN')
    ap.add_argument('--rl_node', type=bool, default=False)
    ap.add_argument('--is_add', type=bool, default=False)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--mode', type=int, default=0)
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--lbd', type=float, default=1.0)
    ap.add_argument('--ssl', type=str, default='dist')

    args = ap.parse_args()
    run_model_DBLP(args)
