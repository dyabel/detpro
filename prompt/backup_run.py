import os, sys
from trainer import test_embedding
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from classname import *
from config import configs
import coop_mini
from trainer import test_embedding, test_embedding_neg
from lr_scheduler import build_lr_scheduler

import random
# torch.random.manual_seed(123)
# random.seed(123)

_, mode, train_dir, val_dir, res_dir, prefix = sys.argv

# Class TODO: compare two ways of preprocessing CLASS_NAME
# CLASS_NAMES_FULL = [x.split('_(')[0] for x in CLASS_NAMES_FULL]
CLASS_NAMES_FULL = [x.replace('(','').replace(')','').replace('_', ' ') for x in CLASS_NAMES_FULL]
CLASS_NAMES_FULL += ["background"]

CLASS_NAMES = []
for id in lvis_base_label_ids:
    CLASS_NAMES.append(CLASS_NAMES_FULL[id])

mapping = [0] * 1203
for i, id in enumerate(lvis_base_label_ids):
    mapping[id] = i
mapping = [866] + mapping
# print(mapping)
mapping = torch.Tensor(mapping).long()

# Data
def load_data(thr_low = 0, thr_up = 1.1):
    feat, label, ious = torch.load(os.path.join(train_dir, "train_data.pth"))
    train_data = []
    for x, y, z in zip(feat, label, ious):
        if thr_low <= z and z < thr_up:
            train_data.append((x, lvis_base_label_ids.index(y.item()), z))
    feat, label, ious = torch.load(os.path.join(val_dir, "val_data.pth"))
    val_data_base = []
    val_data_novel = []
    for x, y, z in zip(feat, label, ious):
        if z >= -0.6:
            if y.item() in lvis_base_label_ids:
                val_data_base.append((x, y, z))
            else:
                val_data_novel.append((x, y, z))
    return train_data, val_data_base, val_data_novel

def load_data_new(data_path : str, remap = False):
    features, labels, ious = torch.load(data_path)
    base = labels.new_zeros(labels.shape).bool()
    novel = labels.new_zeros(labels.shape).bool()
    for i in lvis_base_label_ids: base.logical_or_(labels == i)
    for i in lvis_novel_label_ids: novel.logical_or_(labels == i)
    neg = (labels < 0)
    
    if remap:
        mapping = torch.Tensor(lvis_base_label_ids).long()
        base_labels = (labels[base].view(-1,1)==mapping).int().argmax(dim=1)
        base_data = features[base], base_labels, ious[base]
        neg_data = features[neg], (labels).new_ones(labels[neg].shape)*866, ious[neg]
    else:
        base_data = features[base], labels[base], ious[base]
        neg_data = features[neg], labels[neg], ious[neg]
    novel_data = features[novel], labels[novel], ious[novel]
    
    return base_data, novel_data, neg_data

def data_iou_filter(data, lb, ub):
    features, labels, ious = data
    valid = torch.logical_and(lb <= ious, ious < ub)
    return features[valid], labels[valid], ious[valid]

# train_base, _, train_neg = load_data_new('data/train/train_data.pth', True)
# train_base = data_iou_filter(train_base, 0.5, 1.1)
# train_neg = data_iou_filter(train_neg, 0.1, 1.1)
# val_base, val_novel,val_neg = load_data_new('data/val/val_data.pth')

# val_base = data_iou_filter(val_base, 0.8, 1.1)
# val_neg = data_iou_filter(val_neg, 0.1, 1.1)

def get_freq(data):
    freq = [0] * 1204
    for feat, label, iou in data:
        freq[label] += 1
    return freq

def load_ens_embedding(name_list, norm = False):
    emb = [torch.load(name, map_location='cuda').float() for name in name_list]
    emb = [x / x.norm(dim=-1, keepdim = True) for x in emb]
    emb = sum(emb)
    emb.squeeze_(dim = 0)
    if norm:
        emb = emb / emb.norm(dim = -1, keepdim = True)
    return emb



def test_neg(embedding):
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        test_embedding_neg(embedding, neg_val_dl, thr)


# Clip
clip_model = coop_mini.load_clip_to_cpu().float()
for params in clip_model.parameters():
    params.requires_grad_(False)


# from trainer import get_embedding, checkpoint
# def load_classname_gen_embedding(prompt, classnames):
#     model = coop_mini.CustomCLIP(CLASS_NAMES, clip_model, False, prompt).to('cuda')
#     checkpoint(model, 'ckpt/obj365/pe9', classnames)
# prompt = torch.load('checkpoints/gen6_ens/pos9epoch6_prompt.pth')
# load_classname_gen_embedding(prompt, Objects365_CLASSES)
# quit()

model = coop_mini.CustomCLIP(CLASS_NAMES, clip_model, True).to('cuda')
print('MODEL BUILD COMPLETE')
# total_trainable_params = sum(
#     p.numel() for p in model.parameters() if p.requires_grad)
# print(f'{total_trainable_params:,} training parameters.')


def build_ds(data, lb = -1, ub = 2, remap = False):
    feats, labels, ious = data
    valid = torch.logical_and(lb <= ious, ious < ub)
    feats, labels, ious = feats[valid], labels[valid], ious[valid]
    if remap:
        labels = mapping[labels+1]
    return TensorDataset(feats, labels, ious)

train_data1 = torch.load(os.path.join(train_dir, "train_data.pth"))
val_data1 = torch.load(os.path.join(val_dir, "val_data.pth"))
train_dataset1 = build_ds(train_data1, 0.8, 1.1, remap = True)
train_dl1 = DataLoader(ConcatDataset([train_dataset1]), batch_size = 512, shuffle=True)


# train_data, val_data_base, val_data_novel = load_data(0.8, 1.1)
train_data, val_data_base, val_data_novel = load_data(0.8, 1.6)
# train_data_06, _, _ = load_data(0.0, 0.6)
# train_data_06 = [(x, 866) for x, y in train_data_06]

# Tmp neg data for test
neg_data_raw = torch.load("data/neg_data.pth")
neg_data = [(x, 866, z) for x, y, z in zip(*neg_data_raw) if z > 0.1 and z < 0.51]
# neg_data += train_data_06
neg_val_data = [(x, 866, z) for x, y, z in zip(*neg_data_raw)]
neg_dl = DataLoader(neg_data, batch_size = 1024, shuffle=True)
neg_val_dl = DataLoader(neg_val_data, batch_size = 1024, shuffle=True)

print(len(train_data), len(neg_data))
# train_data = random.sample(train_data, 5000)
# neg_data = random.sample(neg_data, len(train_data))

print(len(train_data), len(neg_data))
if mode != 'test':
    train_data += neg_data
    ...
freq = get_freq(train_data)
freq = [x / len(train_data) * 866 for x in freq]
freq = freq[:866]
freq.append(2 * len(neg_data) / len(train_data)) # background

train_dl = DataLoader(train_data, batch_size = 512, shuffle=True)
val_dl1 = DataLoader(val_data_base, batch_size = 1024, shuffle=True)
val_dl2 = DataLoader(val_data_novel, batch_size = 1024, shuffle=True)


if mode=='test':
    # Ensemble embedding
    # names = ['lvis_text_embedding.pt']
    # names = ['checkpoints/gen6_ens/plain_t001_data2_5e5epoch6.pth']
    # names = [f'checkpoints/gen6_ens/pos{i}epoch6.pth' for i in [5,6,7,8,9]]
    names = [f'ckpt/obj365/pe{i}.pth' for i in [5,6,7,8,9]]
    ensemble_embedding = load_ens_embedding(names, norm = True)# / 5#[:1203]
    torch.save(ensemble_embedding, os.path.join(res_dir, prefix+"_ens.pth"))
    # print("loaded ensemble embedding :", ensemble_embedding.shape)
    test_embedding(ensemble_embedding[lvis_base_label_ids], train_dl)
    test_embedding(ensemble_embedding, val_dl1)
    test_embedding(ensemble_embedding, val_dl2)
    # exit(0)

    test_neg(ensemble_embedding)
    ensemble_embedding = ensemble_embedding[:1203]
    # print("loaded ensemble embedding :", ensemble_embedding.shape)
    test_embedding(ensemble_embedding[lvis_base_label_ids], train_dl)
    test_embedding(ensemble_embedding, val_dl1)
    test_embedding(ensemble_embedding, val_dl2)
    # exit(0)

    test_neg(ensemble_embedding)
    quit()

from torch.optim import SGD, Adam
from trainer import train_epoch, get_embedding, test_embedding, checkpoint
emb = get_embedding(model, CLASS_NAMES_FULL)
test_embedding(emb, val_dl1)
test_embedding(emb, val_dl2)
test_neg(emb)
os.makedirs(res_dir, exist_ok=True)
# checkpoint(model, os.path.join(res_dir, prefix+"epoch0"))

optimizer = SGD(model.prompt_learner.parameters(), lr=2e-3)
scheduler = build_lr_scheduler(optimizer, 6, 0, 1e-5)

for i in range(6):
    print(f"epoch{i+1}")
    train_epoch(model, optimizer, train_dl, freq)
    emb = get_embedding(model, CLASS_NAMES_FULL)
    test_embedding(emb, val_dl1)
    test_embedding(emb, val_dl2)
    test_neg(emb)
    # test_embedding(emb[:1203], val_dl1)
    # test_embedding(emb[:1203], val_dl2)
    # test_neg(emb[:1203])
    scheduler.step()
    checkpoint(model, os.path.join(res_dir, prefix+f"epoch{i+1}"), CLASS_NAMES_FULL)
    checkpoint(model, os.path.join(res_dir, prefix+f"epoch{i+1}_empty"), [""])

checkpoint(model, os.path.join(res_dir, prefix+f"epoch{i+1}"), CLASS_NAMES_FULL)
checkpoint(model, os.path.join(res_dir, prefix+f"coco"), COCO_CLASSES)
checkpoint(model, os.path.join(res_dir, prefix+f"voc"), VOC_CLASSES)
