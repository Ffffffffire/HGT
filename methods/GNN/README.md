# GCN and GAT for benchmark

(To be tuned)

```
python run.py --dataset DBLP --model-type gat
python run.py --dataset DBLP --model-type gcn --weight-decay 1e-6 --lr 1e-3

python run.py --dataset ACM --model-type gat --feats-type 2
python run.py --dataset ACM --model-type gcn --weight-decay 1e-6 --lr 1e-3 --feats-type=0

python run.py --dataset Freebase --model-type gat
python run.py --dataset Freebase --model-type gcn

python run_multi.py --dataset IMDB --model-type gat --feats-type 0 --num-layers 4
python run_multi.py --dataset IMDB --model-type gcn --feats-type 0 --num-layers 3
```
## running environment

* torch 1.6.0 cuda 10.1
* dgl 0.4.3 cuda 10.1
* networkx 2.3
* scikit-learn 0.23.2
* scipy 1.5.2

# GCN without bias:
1
Micro-f1: 0.9407, std: 0.0047
Macro-f1: 0.9319, std: 0.0068
2
Micro-f1: 0.9399, std: 0.0062
Macro-f1: 0.9311, std: 0.0083

# GCN wiht bias:
1
Micro-f1: 0.9457, std: 0.0034
Macro-f1: 0.9382, std: 0.0040
2
Micro-f1: 0.9399, std: 0.0055
Macro-f1: 0.9310, std: 0.0059

# GAT without bias:
1
Micro-f1: 0.9605, std: 0.0062
Macro-f1: 0.9537, std: 0.0080
2
Micro-f1: 0.9556, std: 0.0102
Macro-f1: 0.9477, std: 0.0136
3
Micro-f1: 0.9621, std: 0.0054
Macro-f1: 0.9557, std: 0.0080

