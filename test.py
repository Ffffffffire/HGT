import numpy as np

from scripts.data_loader import data_loader

dl = data_loader('D:/pythoncode/HGB-master/NC/benchmark/data/DBLP')
print(dl.nodes)
print(dl.links)
print(dl.labels_train)
print(dl.labels_train['data'][dl.labels_train['mask']])
pred = dl.labels_test['data'][dl.labels_test['mask']]
print(dl.evaluate(pred))
train_idx = np.nonzero(dl.labels_train['mask'])[0]
test_idx = np.nonzero(dl.labels_test['mask'])[0]
print(train_idx)
print(train_idx.shape)
print(test_idx)
print(test_idx.shape)

meta = [(0, 1), (1, 0)]
print(dl.get_meta_path(meta))
print(dl.get_full_meta_path(meta)[0])

print("dl.nodes['total']:", dl.nodes['total'])
print("dl.links['count']:", dl.links['count'])

for i in range(len(dl.nodes['count'])):
      th = dl.nodes['attr'][i]
      if th is not None:
            print(th.shape)
