# new baseline for benchmark

For message passing with relation attention version:

```
python run_new.py --dataset DBLP
python run_new.py --dataset ACM --feats-type 2
python run_multi.py --dataset IMDB --feats-type 0
python run_new.py --dataset Freebase
```

## running environment

* torch 1.6.0 cuda 10.1
* dgl 0.4.3 cuda 10.1
* networkx 2.3
* scikit-learn 0.23.2
* scipy 1.5.2

# hgt

64 / 8 / 200 / 2 / 1e-3 / 0.5 / 1e-5 / 30 / 1 / False
Epoch 00284 | Train_Loss: 0.0175 | Time: 0.0549
{'micro-f1': 0.5432098765432098, 'macro-f1': 0.5400015976170787}
Epoch 00284 | Val_Loss 3.3346 | Time(s) 0.0140
EarlyStopping counter: 200 out of 200
Early stopping!
{'micro-f1': 0.3292181069958848, 'macro-f1': 0.1238390092879257}

64 / 8 / 200 / 2 / 1e-3 / 0.5 / 1e-5 / 50 / 1 / False
Epoch 00285 | Train_Loss: 0.0112 | Time: 0.0738
{'micro-f1': 0.588477366255144, 'macro-f1': 0.5955641013467534}
Epoch 00285 | Val_Loss 2.8953 | Time(s) 0.0170
EarlyStopping counter: 200 out of 200
Early stopping!
{'micro-f1': 0.5308641975308642, 'macro-f1': 0.4521559298787021}

64 / 8 / 200 / 2 / 1e-3 / 0.5 / 1e-5 / 80 / 1 / False
Epoch 00277 | Train_Loss: 0.0106 | Time: 0.1297
{'micro-f1': 0.7489711934156378, 'macro-f1': 0.7466524449829438}
Epoch 00277 | Val_Loss 1.7589 | Time(s) 0.0229
EarlyStopping counter: 200 out of 200
Early stopping!
{'micro-f1': 0.42386831275720166, 'macro-f1': 0.2999585406301824}

64 / 8 / 200 / 2 / 1e-3 / 0.5 / 1e-5 / 100 / 1 / False
{'micro-f1': 80.7, 'macro-f1': 81.2}

64 / 8 / 200 / 2 / 1e-3 / 0.5 / 0 / 100 / 1 / False
Epoch 00299 | Train_Loss: 0.0065 | Time: 0.1695
{'micro-f1': 0.8271604938271605, 'macro-f1': 0.8231952739166946}
Epoch 00299 | Val_Loss 1.3540 | Time(s) 0.0209
EarlyStopping counter: 200 out of 200
Early stopping!
{'micro-f1': 0.7983539094650206, 'macro-f1': 0.793405539088105}

64 / 8 / 500 / 2 / 1e-3 / 0.5 / 0 / 100 / 1 / False
Epoch 00569 | Train_Loss: 0.0014 | Time: 0.1835
{'micro-f1': 0.8106995884773662, 'macro-f1': 0.8097448120426468}
Epoch 00569 | Val_Loss 1.4770 | Time(s) 0.0219
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.7366255144032922, 'macro-f1': 0.7301599511599511}

64 / 8 / 500 / 2 / 1e-3 / 0.5 / 0 / 100 / 1 / True
Epoch 00544 | Train_Loss: 0.0006 | Time: 0.1745
{'micro-f1': 0.757201646090535, 'macro-f1': 0.7463117800681892}
Epoch 00544 | Val_Loss 0.8682 | Time(s) 0.0229
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.7695473251028806, 'macro-f1': 0.7652350894437557}

64 / 8 / 500 / 2 / 1e-3 / 0 / 0 / 100 / 1 / False
Epoch 00549 | Train_Loss: 0.3422 | Time: 0.1686
{'micro-f1': 0.6378600823045267, 'macro-f1': 0.5436018352409627}
Epoch 00549 | Val_Loss 1.2749 | Time(s) 0.0219
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.6460905349794238, 'macro-f1': 0.5413116893084657}

64 / 8 / 500 / 2 / 1e-3 / 0.1 / 0 / 100 / 1 / False
Epoch 00540 | Train_Loss: 0.0007 | Time: 0.1877
{'micro-f1': 0.8065843621399177, 'macro-f1': 0.8065932060894347}
Epoch 00540 | Val_Loss 1.1459 | Time(s) 0.0249
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8189300411522634, 'macro-f1': 0.8184517108877509}

64 / 8 / 500 / 2 / 1e-3 / 0.2 / 0 / 100 / 1 / False
Epoch 00533 | Train_Loss: 0.0009 | Time: 0.1775
{'micro-f1': 0.7860082304526749, 'macro-f1': 0.7825495176518884}
Epoch 00533 | Val_Loss 1.3208 | Time(s) 0.0219
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8024691358024691, 'macro-f1': 0.7982103980577049}

64 / 8 / 500 / 2 / 1e-3 / 0.3 / 0 / 100 / 1 / False
{'micro-f1': 0.7860082304526749, 'macro-f1': 0.7812518014408105}
Epoch 00550 | Val_Loss 1.4519 | Time(s) 0.0229
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.7942386831275721, 'macro-f1': 0.7930413168340786}

64 / 8 / 500 / 2 / 1e-3 / 0.4 / 0 / 100 / 1 / False
Epoch 00585 | Train_Loss: 0.0010 | Time: 0.1775
{'micro-f1': 0.7818930041152263, 'macro-f1': 0.7779568790235393}
Epoch 00585 | Val_Loss 1.6990 | Time(s) 0.0269
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8600823045267489, 'macro-f1': 0.8601314573551485}

64 / 8 / 500 / 2 / 1e-3 / 0.8 / 0 / 100 / 1 / False
Epoch 00999 | Train_Loss: 1.3780 | Time: 0.1905
{'micro-f1': 0.26337448559670784, 'macro-f1': 0.10423452768729642}
Epoch 00999 | Val_Loss 1.3901 | Time(s) 0.0230
EarlyStopping counter: 379 out of 500
{'micro-f1': 0.26337448559670784, 'macro-f1': 0.10423452768729642}

64 / 4 / 500 / 2 / 1e-3 / 0.4 / 0 / 100 / 1 / False
Epoch 00615 | Train_Loss: 0.0008 | Time: 0.1247
{'micro-f1': 0.9053497942386831, 'macro-f1': 0.8956161402444232}
Epoch 00615 | Val_Loss 0.6708 | Time(s) 0.0180
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8847736625514403, 'macro-f1': 0.8776532111787003}

64 / 2 / 500 / 2 / 1e-3 / 0.4 / 0 / 100 / 1 / False
{'micro-f1': 0.8847736625514403, 'macro-f1': 0.879529831659376}
Epoch 00579 | Val_Loss 0.8050 | Time(s) 0.0150
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8806584362139918, 'macro-f1': 0.8736730703416782}

64 / 1 / 500 / 2 / 1e-3 / 0.4 / 0 / 100 / 1 / False
Epoch 00570 | Train_Loss: 0.0010 | Time: 0.0858
{'micro-f1': 0.8106995884773662, 'macro-f1': 0.8067456110519844}
Epoch 00570 | Val_Loss 1.4472 | Time(s) 0.0189
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8065843621399177, 'macro-f1': 0.8051213341884225}

64 / 16 / 500 / 2 / 1e-3 / 0.4 / 0 / 100 / 1 / False
Epoch 00532 | Train_Loss: 0.0124 | Time: 0.2942
{'micro-f1': 0.5761316872427984, 'macro-f1': 0.5599006648464502}
Epoch 00532 | Val_Loss 3.2247 | Time(s) 0.0342
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.522633744855967, 'macro-f1': 0.4270000321866121}

64 / 4 / 500 / 1 / 1e-3 / 0.4 / 0 / 100 / 1 / False
Epoch 00599 | Train_Loss: 0.0008 | Time: 0.0648
{'micro-f1': 0.8724279835390947, 'macro-f1': 0.8666513673338749}
Epoch 00599 | Val_Loss 0.8062 | Time(s) 0.0150
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8683127572016461, 'macro-f1': 0.8615962523415653}

64 / 4 / 500 / 3 / 1e-3 / 0.4 / 0 / 100 / 1 / False
Epoch 00559 | Train_Loss: 0.0010 | Time: 0.1745
{'micro-f1': 0.8106995884773662, 'macro-f1': 0.8202515257710064}
Epoch 00559 | Val_Loss 1.2909 | Time(s) 0.0229
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8806584362139918, 'macro-f1': 0.8809094311703007}

64 / 4 / 500 / 4 / 1e-3 / 0.4 / 0 / 100 / 1 / False
{'micro-f1': 0.7613168724279835, 'macro-f1': 0.7483649383787578}
Epoch 00610 | Val_Loss 2.2545 | Time(s) 0.0310
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8971193415637861, 'macro-f1': 0.884076148124942}

32 / 4 / 500 / 2 / 1e-3 / 0.4 / 0 / 100 / 1 / False
Epoch 00577 | Train_Loss: 0.0047 | Time: 0.1137
{'micro-f1': 0.9094650205761317, 'macro-f1': 0.905775931241488}
Epoch 00577 | Val_Loss 0.6141 | Time(s) 0.0200
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.9176954732510288, 'macro-f1': 0.9145991313737134}

16 / 4 / 500 / 2 / 1e-3 / 0.4 / 0 / 100 / 1 / False
Epoch 00661 | Train_Loss: 0.0514 | Time: 0.1067
{'micro-f1': 0.7613168724279835, 'macro-f1': 0.7652220173074195}
Epoch 00661 | Val_Loss 1.4956 | Time(s) 0.0199
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.7283950617283951, 'macro-f1': 0.7246323415104313}

128 / 4 / 500 / 2 / 1e-3 / 0.4 / 0 / 100 / 1 / False
Epoch 00524 | Train_Loss: 0.0003 | Time: 0.1785
{'micro-f1': 0.9053497942386831, 'macro-f1': 0.9020555459720546}
Epoch 00524 | Val_Loss 0.8152 | Time(s) 0.0239
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.9094650205761317, 'macro-f1': 0.8976947569788734}

without FFN:
Epoch 00546 | Train_Loss: 0.0021 | Time: 0.0977
{'micro-f1': 0.8477366255144033, 'macro-f1': 0.8457982457982458}
Epoch 00546 | Val_Loss 1.0090 | Time(s) 0.0160
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8395061728395061, 'macro-f1': 0.8381650185370427}

without FFN dropout 0:
Epoch 00545 | Train_Loss: 0.3378 | Time: 0.0898
{'micro-f1': 0.5102880658436214, 'macro-f1': 0.4325602068427448}
Epoch 00545 | Val_Loss 1.8886 | Time(s) 0.0140
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.5679012345679012, 'macro-f1': 0.4830391052817523}

without FFN dropout 0.1:
Epoch 00539 | Train_Loss: 0.0009 | Time: 0.0888
{'micro-f1': 0.8148148148148148, 'macro-f1': 0.8129898098402036}
Epoch 00539 | Val_Loss 0.7842 | Time(s) 0.0140
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8395061728395061, 'macro-f1': 0.831827576979772}

without FFN 3 layers:
Epoch 00560 | Train_Loss: 0.0024 | Time: 0.1376
{'micro-f1': 0.757201646090535, 'macro-f1': 0.7493691213613908}
Epoch 00560 | Val_Loss 1.7747 | Time(s) 0.0189
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.7860082304526749, 'macro-f1': 0.7733649501915563}

withou FFN 32 hidden_dim:
Epoch 00599 | Train_Loss: 0.0070 | Time: 0.0808
{'micro-f1': 0.8806584362139918, 'macro-f1': 0.8647094564506332}
Epoch 00599 | Val_Loss 0.6254 | Time(s) 0.0160
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8930041152263375, 'macro-f1': 0.8834260726835843}

without layer-norm
Epoch 00557 | Train_Loss: 0.0020 | Time: 0.0778
{'micro-f1': 0.6707818930041153, 'macro-f1': 0.668262772307394}
Epoch 00557 | Val_Loss 3.1140 | Time(s) 0.0120
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.6625514403292181, 'macro-f1': 0.6019453656119268}

without layer-norm dropout 0 无法训练

without layer-norm dropout 0.1
Epoch 00543 | Train_Loss: 0.0001 | Time: 0.0738
{'micro-f1': 0.5308641975308642, 'macro-f1': 0.476393146040361}
Epoch 00543 | Val_Loss 5.4835 | Time(s) 0.0130
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.5267489711934157, 'macro-f1': 0.49678833071690215}

without layer-norm 3-num-layers
Epoch 00526 | Train_Loss: 0.0000 | Time: 0.1007
{'micro-f1': 0.43209876543209874, 'macro-f1': 0.4125616266539084}
Epoch 00526 | Val_Loss 9.2682 | Time(s) 0.0140
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.5967078189300411, 'macro-f1': 0.5073586338325939}

without layer-norm 64-hidden:
Epoch 00534 | Train_Loss: 0.0022 | Time: 0.1017
{'micro-f1': 0.7037037037037037, 'macro-f1': 0.7162817560746044}
Epoch 00534 | Val_Loss 3.4191 | Time(s) 0.0170
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.6748971193415638, 'macro-f1': 0.6873551693404634}

without layer-norm 128-hidden
Epoch 00529 | Train_Loss: 0.0000 | Time: 0.1402
{'micro-f1': 0.6666666666666666, 'macro-f1': 0.6842620742917785}
Epoch 00529 | Val_Loss 5.6292 | Time(s) 0.0215
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.7942386831275721, 'macro-f1': 0.7881554931270541}

1-FFN:
Epoch 00520 | Train_Loss: 0.0004 | Time: 0.1616
{'micro-f1': 0.8971193415637861, 'macro-f1': 0.8891766248323625}
Epoch 00520 | Val_Loss 0.7784 | Time(s) 0.0209
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8888888888888888, 'macro-f1': 0.8792272155005014}

no-residual 32:
Epoch 00663 | Train_Loss: 0.0064 | Time: 0.1058
{'micro-f1': 0.6707818930041153, 'macro-f1': 0.6799743415328989}
Epoch 00663 | Val_Loss 2.5000 | Time(s) 0.0169
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8024691358024691, 'macro-f1': 0.7999843947774522}

randomwalk + local neighborhood:
Epoch 00604 | Train_Loss: 0.0080 | Time: 0.1117
{'micro-f1': 0.8641975308641975, 'macro-f1': 0.8544913287128184}
Epoch 00604 | Val_Loss 0.9554 | Time(s) 0.0180
EarlyStopping counter: 500 out of 500
Early stopping!
{'micro-f1': 0.8765432098765432, 'macro-f1': 0.8718403242542034}








5 run:

32 / 4 / 500 / 2 / 1e-3 / 0.4 / 0 / 100 / 1 / False

Micro-f1: 0.8757, std: 0.0141
Macro-f1: 0.8706, std: 0.0171

32 / 4 / 500 / 2 / 1e-3 / 0.5 / 0 / 100 / 1 / False

Micro-f1: 0.9251, std: 0.0185
Macro-f1: 0.9191, std: 0.0182

32 / 4 / 500 / 2 / 1e-3 / 0.6 / 0 / 100 / 1 / False

Micro-f1: 0.8856, std: 0.0107
Macro-f1: 0.8817, std: 0.0094

32 / 4 / 500 / 2 / 1e-3 / 0 / 0 / 100 / 1 / False

Micro-f1: 0.6601, std: 0.0939
Macro-f1: 0.5987, std: 0.1708

64 / 4 / 500 / 2 / 1e-3 / 0.5 / 0 / 100 / 1 / False

Micro-f1: 0.9012, std: 0.0097
Macro-f1: 0.8970, std: 0.0103

64 / 4 / 500 / 3 / 1e-3 / 0.5 / 0 / 100 / 1 / False

Micro-f1: 0.8864, std: 0.0161
Macro-f1: 0.8815, std: 0.0179

32 / 4 / 500 / 3 / 1e-3 / 0.5 / 0 / 100 / 1 / False

Micro-f1: 0.8593, std: 0.0328
Macro-f1: 0.8567, std: 0.0312

128 / 4 / 500 / 2 / 1e-3 / 0.5 / 0 / 100 / 1 / False

Micro-f1: 0.8593, std: 0.0328
Macro-f1: 0.8567, std: 0.0312



baseline Simple-HGN

Micro-f1: 0.9103, std: 0.0079
Macro-f1: 0.9073, std: 0.0087

baseline GAT

Micro-f1: 0.9457, std: 0.0107
Macro-f1: 0.9396, std: 0.0123

Micro-f1: 0.9679, std: 0.0034
Macro-f1: 0.9636, std: 0.0042

baseline GCN

Micro-f1: 0.9111, std: 0.0037
Macro-f1: 0.9056, std: 0.0037

500：
Micro-f1: 0.9210, std: 0.0251
Macro-f1: 0.9124, std: 0.0266

200：
Micro-f1: 0.9136, std: 0.0273
Macro-f1: 0.9038, std: 0.0305

seq 200:
Micro-f1: 0.9350, std: 0.0213
Macro-f1: 0.9299, std: 0.0228

64 4 1000 200 3 1e-4 0.5 5e-4 100 GIN 4

64 4 1000 200 3 1e-4 0 0 100 GIN 4

Micro-f1: 0.7728, std: 0.0492
Macro-f1: 0.7646, std: 0.0458

64 4 1000 200 3 1e-4 0.5 5e-4 100 GIN 4


# without bias
1
Micro-f1: 0.9284, std: 0.0075
Macro-f1: 0.9204, std: 0.0100
2
Micro-f1: 0.9235, std: 0.0099
Macro-f1: 0.9154, std: 0.0114
3
Micro-f1: 0.9350, std: 0.0110
Macro-f1: 0.9266, std: 0.0143

# ffn-dim

32
1
Micro-f1: 0.9177, std: 0.0140
Macro-f1: 0.9083, std: 0.0149
2
Micro-f1: 0.9300, std: 0.0157
Macro-f1: 0.9196, std: 0.0172

64
1
Micro-f1: 0.9383, std: 0.0120
Macro-f1: 0.9321, std: 0.0130
2
Micro-f1: 0.9185, std: 0.0074
Macro-f1: 0.9093, std: 0.0091

# lr 
1e-4
Micro-f1: 0.9152, std: 0.0141
Macro-f1: 0.9048, std: 0.0157
5e-4
Micro-f1: 0.9440, std: 0.0055
Macro-f1: 0.9381, std: 0.0064
2
Micro-f1: 0.9267, std: 0.0089
Macro-f1: 0.9172, std: 0.0118
3
Micro-f1: 0.9202, std: 0.0171
Macro-f1: 0.9087, std: 0.0191

# layer-3
1
Micro-f1: 0.9177, std: 0.0065
Macro-f1: 0.9087, std: 0.0063
2
Micro-f1: 0.9029, std: 0.0055
Macro-f1: 0.8918, std: 0.0065

# layer-1
1
Micro-f1: 0.9243, std: 0.0115
Macro-f1: 0.9169, std: 0.0140
2
Micro-f1: 0.9416, std: 0.0045
Macro-f1: 0.9354, std: 0.0044
3
Micro-f1: 0.9202, std: 0.0090
Macro-f1: 0.9139, std: 0.0111

# layer-4
Micro-f1: 0.9177, std: 0.0127
Macro-f1: 0.9068, std: 0.0143

Micro-f1: 0.9243, std: 0.0103
Macro-f1: 0.9154, std: 0.0111

# layer-5
Micro-f1: 0.7012, std: 0.2037
Macro-f1: 0.5852, std: 0.3018

# 64 128
Micro-f1: 0.9259, std: 0.0087
Macro-f1: 0.9157, std: 0.0104

Micro-f1: 0.9259, std: 0.0050
Macro-f1: 0.9179, std: 0.0061

# 64 64
Micro-f1: 0.9358, std: 0.0080
Macro-f1: 0.9279, std: 0.0089
2
Micro-f1: 0.9325, std: 0.0069
Macro-f1: 0.9239, std: 0.0085

# 64 256 seq 120
Micro-f1: 0.9407, std: 0.0069
Macro-f1: 0.9339, std: 0.0066

Micro-f1: 0.9506, std: 0.0041
Macro-f1: 0.9456, std: 0.0038

Micro-f1: 0.9235, std: 0.0111
Macro-f1: 0.9184, std: 0.0114

# 64 256 seq 100
Micro-f1: 0.9276, std: 0.0095
Macro-f1: 0.9178, std: 0.0100

Micro-f1: 0.9202, std: 0.0075
Macro-f1: 0.9097, std: 0.0073

# 128 128 seq 100
Micro-f1: 0.9103, std: 0.0089
Macro-f1: 0.8985, std: 0.0102
1
Micro-f1: 0.9226, std: 0.0074
Macro-f1: 0.9098, std: 0.0087

# 128 256
Micro-f1: 0.9193, std: 0.0125
Macro-f1: 0.9062, std: 0.0153
1
Micro-f1: 0.9210, std: 0.0054
Macro-f1: 0.9120, std: 0.0064

# 32 64 num-heads 8

Micro-f1: 0.8444, std: 0.0084
Macro-f1: 0.8333, std: 0.0101

icro-f1: 0.7844, std: 0.0803
Macro-f1: 0.7443, std: 0.1221

# 32 64 num-heads 2

Micro-f1: 0.9317, std: 0.0085
Macro-f1: 0.9248, std: 0.0111

Micro-f1: 0.9399, std: 0.0085
Macro-f1: 0.9311, std: 0.0098

# 32 64 num-heads 1
Micro-f1: 0.8790, std: 0.0161
Macro-f1: 0.8630, std: 0.0193

# 32 64 num-heads 4

Micro-f1: 0.9366, std: 0.0150
Macro-f1: 0.9291, std: 0.0154

Micro-f1: 0.9152, std: 0.0075
Macro-f1: 0.9054, std: 0.0080

# 32 64 with bias:
Micro-f1: 0.9062, std: 0.0121
Macro-f1: 0.8954, std: 0.0138

# 32 64 seq 120

Micro-f1: 0.9424, std: 0.0127
Macro-f1: 0.9355, std: 0.0151

# 32 64 seq 140
1
Micro-f1: 0.9300, std: 0.0105
Macro-f1: 0.9201, std: 0.0122
2
Micro-f1: 0.9407, std: 0.0090
Macro-f1: 0.9332, std: 0.0109
3
Micro-f1: 0.9300, std: 0.0148
Macro-f1: 0.9235, std: 0.0170

# 32 64 seq 150
1
Micro-f1: 0.9547, std: 0.0105
Macro-f1: 0.9490, std: 0.0125
2
Micro-f1: 0.9284, std: 0.0132
Macro-f1: 0.9200, std: 0.0141
3
Micro-f1: 0.9358, std: 0.0111
Macro-f1: 0.9268, std: 0.0136

# 32 64 seq 180
1
Micro-f1: 0.9523, std: 0.0047
Macro-f1: 0.9487, std: 0.0051
2
Micro-f1: 0.9580, std: 0.0079
Macro-f1: 0.9544, std: 0.0075
3
Micro-f1: 0.9580, std: 0.0098
Macro-f1: 0.9535, std: 0.0106

# 32 64 seq 200
1
Micro-f1: 0.9391, std: 0.0107
Macro-f1: 0.9327, std: 0.0126
2
Micro-f1: 0.9399, std: 0.0111
Macro-f1: 0.9334, std: 0.0138

# 32 64 seq 180 without FFN-bias
Micro-f1: 0.9506, std: 0.0071
Macro-f1: 0.9455, std: 0.0080

Micro-f1: 0.9556, std: 0.0107
Macro-f1: 0.9522, std: 0.0116

# 32 64 seq 200 without FFN-bias
Micro-f1: 0.9556, std: 0.0107
Macro-f1: 0.9522, std: 0.0116

Micro-f1: 0.9457, std: 0.0079
Macro-f1: 0.9397, std: 0.0077

# 32 64 seq 180 GIN K 2 

Micro-f1: 0.9235, std: 0.0161
Macro-f1: 0.9143, std: 0.0173

# 32 64 seq 180 GIN K 1

Micro-f1: 0.9259, std: 0.0224
Macro-f1: 0.9184, std: 0.0249

# 32 64 seq 180 GIN K 3

Micro-f1: 0.9440, std: 0.0037
Macro-f1: 0.9372, std: 0.0049

接近

# 32 64 seq 180 GIN K 4

Micro-f1: 0.9440, std: 0.0037
Macro-f1: 0.9372, std: 0.0049

# 32 64 seq 180 GIN K 5

Micro-f1: 0.9481, std: 0.0099
Macro-f1: 0.9420, std: 0.0108

# 32 64 seq 180 GIN K 6

Micro-f1: 0.9523, std: 0.0103
Macro-f1: 0.9453, std: 0.0114

# 32 64 seq 180 GIN K 7
Micro-f1: 0.9556, std: 0.0074
Macro-f1: 0.9502, std: 0.0087

# 32 64 seq 180 GIN K 8
Micro-f1: 0.9416, std: 0.0079
Macro-f1: 0.9354, std: 0.0064

# 32 64 seq 180 GCN K 1
Micro-f1: 0.8979, std: 0.0168
Macro-f1: 0.8854, std: 0.0192

# 32 64 seq 180 GCN K 2
Micro-f1: 0.9070, std: 0.0207
Macro-f1: 0.8929, std: 0.0266

# 32 64 seq 180 SAGE K 1

Micro-f1: 0.8922, std: 0.0114
Macro-f1: 0.8770, std: 0.0169

# 32 64 seq 180 SAGE K 2

Micro-f1: 0.9407, std: 0.0062
Macro-f1: 0.9324, std: 0.0082

# 32 64 seq 180 SAGE K 3

Micro-f1: 0.9053, std: 0.0151
Macro-f1: 0.8926, std: 0.0177

# 32 64 seq 180 SAGE K 4

Micro-f1: 0.9012, std: 0.0162
Macro-f1: 0.8906, std: 0.0140

# 32 64 seq 180 SAGE K 5

Micro-f1: 0.9004, std: 0.0335
Macro-f1: 0.8912, std: 0.0324





# Attention Map K 7 
Micro-f1: 0.9391, std: 0.0034
Macro-f1: 0.9332, std: 0.0033
6
Micro-f1: 0.9342, std: 0.0116
Macro-f1: 0.9252, std: 0.0125
5
Micro-f1: 0.9547, std: 0.0065
Macro-f1: 0.9482, std: 0.0066
4
Micro-f1: 0.9325, std: 0.0085
Macro-f1: 0.9238, std: 0.0084
3
Micro-f1: 0.9531, std: 0.0085
Macro-f1: 0.9490, std: 0.0108
2
Micro-f1: 0.9416, std: 0.0094
Macro-f1: 0.9360, std: 0.0088
1
Micro-f1: 0.9317, std: 0.0080
Macro-f1: 0.9212, std: 0.0076

# Node-wise K 2

Micro-f1: 0.9572, std: 0.0062
Macro-f1: 0.9513, std: 0.0077

Micro-f1: 0.9556, std: 0.0054
Macro-f1: 0.9523, std: 0.0063

# K 3

Micro-f1: 0.9523, std: 0.0047
Macro-f1: 0.9462, std: 0.0052   

# K 4

Micro-f1: 0.9325, std: 0.0095
Macro-f1: 0.9221, std: 0.0124

# K 1 

Micro-f1: 0.9490, std: 0.0107
Macro-f1: 0.9444, std: 0.0123

# bias = False K 2

Micro-f1: 0.9374, std: 0.0061
Macro-f1: 0.9305, std: 0.0077

# Normalize = False K 2

Micro-f1: 0.9638, std: 0.0054
Macro-f1: 0.9609, std: 0.0065

# K = 1

Micro-f1: 0.9490, std: 0.0085
Macro-f1: 0.9421, std: 0.0095

# K = 3

Micro-f1: 0.9424, std: 0.0058
Macro-f1: 0.9378, std: 0.0054

# rl_dim 8

Micro-f1: 0.9457, std: 0.0054
Macro-f1: 0.9395, std: 0.0055

# rl_dim 4

Micro-f1: 0.9481, std: 0.0111
Macro-f1: 0.9418, std: 0.0134

Micro-f1: 0.9481, std: 0.0125
Macro-f1: 0.9426, std: 0.0134

# normalize
Micro-f1: 0.9465, std: 0.0058
Macro-f1: 0.9387, std: 0.0066

# rl_dim 8 

Micro-f1: 0.9531, std: 0.0023
Macro-f1: 0.9475, std: 0.0022

# rl_dim 16

Micro-f1: 0.9391, std: 0.0054
Macro-f1: 0.9334, std: 0.0059

# 180  K 2 normalize = False

Micro-f1: 0.9424, std: 0.0087
Macro-f1: 0.9361, std: 0.0075

# 180 K 3 normalize = False

Micro-f1: 0.9531, std: 0.0080
Macro-f1: 0.9483, std: 0.0081

# 180 K 4 normalize = False

Micro-f1: 0.9465, std: 0.0029
Macro-f1: 0.9396, std: 0.0028

# 180 K 1 normalize = False

Micro-f1: 0.9449, std: 0.0069
Macro-f1: 0.9389, std: 0.0072

# 150 K 3 normalize = False

Micro-f1: 0.9506, std: 0.0058
Macro-f1: 0.9463, std: 0.0073

# 150 K 2 normalize = False
Micro-f1: 0.9325, std: 0.0062
Macro-f1: 0.9249, std: 0.0074

# 150 K 4 normalize = False
Micro-f1: 0.9449, std: 0.0069
Macro-f1: 0.9394, std: 0.0068

# 120 K 3 normalize = False
Micro-f1: 0.9309, std: 0.0135
Macro-f1: 0.9248, std: 0.0150

# 120 K 2 normalize = False
Micro-f1: 0.9210, std: 0.0107
Macro-f1: 0.9125, std: 0.0131

# 120 K 4 normalize = False
Micro-f1: 0.9473, std: 0.0089
Macro-f1: 0.9387, std: 0.0094

# 100 K 4 normalize = False
Micro-f1: 0.9267, std: 0.0079
Macro-f1: 0.9184, std: 0.0088

# 100 K 3 normalize = False
Micro-f1: 0.9251, std: 0.0135
Macro-f1: 0.9196, std: 0.0148

# 100 K 2 normalize = False
Micro-f1: 0.9193, std: 0.0085
Macro-f1: 0.9117, std: 0.0104

# 100 K 1 normalize = False
Micro-f1: 0.9325, std: 0.0099
Macro-f1: 0.9225, std: 0.0112
