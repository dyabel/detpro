import os, sys
import torch

path = sys.argv[1]
save_name = os.path.join(path, sys.argv[2])

if os.path.exists(save_name):
    print('Data: target already exists!')
    exit(0)

feats = []
labels = []
ious = []
files = []
for splt in os.listdir(path):
    print(splt)
    files += [os.path.join(path, splt, f) for f in os.listdir(os.path.join(path, splt))]
print('total', len(files), 'files')

for pth in files:
    feat, label, iou = torch.load(pth)
    print(len(feat), len(label),len(iou))
    iou = torch.cat([iou, iou.new_ones(len(label) - len(iou))]) # fix a bug in collect iou
    feats.append(feat)
    labels.append(label)
    ious.append(iou)

feats = torch.cat(feats)
labels = torch.cat(labels)
ious = torch.cat(ious)
print(feats.shape, labels.shape, ious.shape)

torch.save((feats, labels, ious), save_name)
