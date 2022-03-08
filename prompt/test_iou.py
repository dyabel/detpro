import torch
from classname import *
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import os
# name5 = ['checkpoints/iou/pos5epoch6.pth']
# name6 = ['checkpoints/iou/pos6epoch6.pth']
# name7 = ['checkpoints/iou/pos7epoch6.pth']
# name8 = ['checkpoints/iou/pos8epoch6.pth']
# name9 = ['checkpoints/iou/pos9epoch6.pth']
def load_ens_embedding(name_list, norm = False):
    emb = [torch.load(name, map_location='cuda').float() for name in name_list]
    emb = [x / x.norm(dim=-1, keepdim = True) for x in emb]
    emb = sum(emb)
    emb.squeeze_(dim = 0)
    if norm:
        emb = emb / emb.norm(dim = -1, keepdim = True)
    return emb

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

# iou5_embedding = load_ens_embedding(name5, norm = True)# / 5#[:1203]
# iou6_embedding = load_ens_embedding(name6, norm = True)# / 5#[:1203]
# iou7_embedding = load_ens_embedding(name7, norm = True)# / 5#[:1203]
# iou8_embedding = load_ens_embedding(name8, norm = True)# / 5#[:1203]
# iou9_embedding = load_ens_embedding(name9, norm = True)# / 5#[:1203]
device = "cuda" if torch.cuda.is_available() else "cpu"

# iou_embeddings = torch.stack([load_ens_embedding([f'checkpoints/iou/pos{i}epoch6.pth'], norm = True) for i in range(5,10)]).to(device)
# text_embedding = load_ens_embedding([f'lvis_text_embedding.pt'], norm = True).to(device)
text_embedding = load_ens_embedding([f'checkpoints/exp1/soft_epoch5.pth'], norm = True).to(device)
path = 'save_feature/data/lvis_test_iou/train/train2017'
files = []
for file in os.listdir(path):
    print('#'*100)
    proposal_embeddings, labels, ious = torch.load(os.path.join(path,file))
    print(labels,ious.sort()[1])
    proposal_sims = proposal_embeddings.cuda()@text_embedding.T
    tmp = proposal_sims.div(0.01).softmax(dim=1).max(dim=-1)
    print(tmp)
    print(tmp[0].sort()[1])
# print(iou_embeddings.shape)
# val_base, val_novel,val_neg = load_data_new('data/lvis_clip_image_proposal_embedding/val/val_data.pth')
# val_base, val_novel, val_neg = TensorDataset(*val_base), TensorDataset(*val_novel), TensorDataset(*val_neg)
# val_dataset = DataLoader(val_base, batch_size = 1, shuffle=True)
# def check_iou(iou):
#     return torch.floor(iou*10-4)

# for proposal_embedding,label,iou in val_dataset: 
#     iou_group =  check_iou(iou)
#     proposal_embedding = proposal_embedding.to(device)
#     sim = torch.einsum('nc,abc->nab',proposal_embedding,iou_embeddings)
#     # sim = sim.div(0.01).softmax(dim=-1)
#     sim = sim.max(dim=-1)[0].squeeze()
#     print(sim.argmax(dim=-1),iou_group)


