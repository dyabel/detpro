from classname import *
import torch
from torch.nn import functional as F
import time
from lr_scheduler import build_lr_scheduler
from torch.optim import SGD
import random
from config import temperature

from torch.cuda.amp import autocast as autocast

torch.cuda.empty_cache()

def get_embedding(model, class_names):
    with torch.no_grad():
        prompts, tokenized_prompts = model.prompt_learner.forward_for_classes(class_names)
        text_features = model.text_encoder(prompts, tokenized_prompts)

        # text_features = torch.cat([text_features, model.bg_embedding], dim = 0)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

def checkpoint(model, name, class_names):
    with torch.no_grad():
        prompts, tokenized_prompts = model.prompt_learner.forward_for_classes(class_names)
        text_features = model.text_encoder(prompts, tokenized_prompts)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features = torch.cat([text_features, model.bg_embedding], dim=0)

        text_features = text_features[None, ...]
        torch.save(text_features, name+".pth")
        prompt = model.prompt_learner.ctx.data
        torch.save(prompt, name+"_prompt.pth")


def accuracy1(logits, labels):
    return torch.argmax(logits, dim = -1).eq(labels).sum()
def accuracy5(logits, labels):
    return logits.float().topk(5)[1].eq(labels[:,None]).sum()


def test_embedding(embedding, ds):
    iter = 0
    acc1, acc5 = 0, 0
    avg_score, avg_var, entropy = 0, 0, 0

    for feat, label, iou in ds:
        iter += 1
        if iter % 100 == 0:
            print(iter, '/', len(ds), end='\r')

        res = feat.to('cuda') @ embedding.t() / temperature
        res = F.softmax(res, dim=-1)
        # res[:,lvis_base_label_ids] = -1e10
        acc1 += accuracy1(res.cpu(), label)
        acc5 += accuracy5(res.cpu(), label)

        # avg_score += res.max(dim = -1)[0][:1203].sum()
        # avg_var += res.var(dim=-1)[:1203].sum()
        # entropy += -res.log().sum(dim=-1)[:1203].sum()
        res = res[:,:1203]
        avg_score += res.max(dim = -1)[0].sum()
        avg_var += res.var(dim=-1).sum()
        entropy += (-res.log() * res).sum()

    acc1 = acc1.item() / len(ds.dataset)
    acc5 = acc5.item() / len(ds.dataset)
    avg_score = avg_score.item() / len(ds.dataset)
    avg_var = avg_var.item() / len(ds.dataset)
    entropy = entropy.item() / len(ds.dataset)
    print(f"test acc: top1={acc1}   top5={acc5}   total={len(ds.dataset)}")
    print(f"avg_score: {avg_score}      avg_var: {avg_var}     entropy: {entropy}")

def test_embedding_neg(embedding, ds, thr):
    iter = 0
    pos = 0
    avg_score, avg_var, entropy = 0, 0, 0

    for feat, label, iou in ds:
        iter += 1
        if iter % 100 == 0:
            print(iter, '/', len(ds), end='\r')

        res = feat.to('cuda') @ embedding.t() / temperature
        res = F.softmax(res, dim=-1)[:,:1203]
        # print(res)
        pos += (res.max(dim=-1)[0] >= thr).sum()

        avg_score += res.max(dim = -1)[0].sum()
        avg_var += res.var(dim=-1).sum()
        entropy += (-res.log() * res).sum()

    avg_score = avg_score.item() / len(ds.dataset)
    avg_var = avg_var.item() / len(ds.dataset)
    entropy = entropy.item() / len(ds.dataset)
    print(f"test neg(thr={thr}): pos={pos}  total={len(ds.dataset)}")
    print(f"avg_score: {avg_score}      avg_var: {avg_var}     entropy: {entropy}")


def softLabel(label, iou):
    softlb = iou.new_zeros((label.shape[0], 867))
    softlb.scatter_(1, label[:,None].long(), iou[:,None])
    softlb[label!=866,-1] = 1-iou[label!=866]
    softlb[label==866] = 1/867
    return softlb

def softCrossEntropy(logit, target):
    log_likelihood = -F.log_softmax(logit, dim = 1)
    return torch.sum(log_likelihood * target)


def train_epoch(model, optimizer, ds, freq = None, mode = 'hinge'):
    print("train mode :", mode)
    with torch.no_grad():
        emb = model.get_embedding()
        print("embbeding shape :", emb.shape)
    
    acc1, acc5 = 0, 0
    idb = 0
    time_s = time.time()
    with autocast():
        for feat, label, iou in ds:
            idb += 1
            if idb % 100 == 0:
                print(idb, '/', len(ds), end='  ')

            emb = model.get_embedding()[:867]
            emb = emb / emb.norm(dim = -1, keepdim = True)
            # emb = torch.cat([embr[:866] / embr[:866].norm(dim = -1, keepdim = True), embr[-1:]])
            # sim = emb @ emb.T - torch.eye(emb.shape[0]).cuda()
            # print(sim)
            # print(sim.shape)
            # print((sim + torch.eye(emb.shape[0]).cuda()).min(), sim.max(), sim.mean())
            # # input()
            # closs = (sim-0.8).maximum(torch.tensor(0).cuda()).sum()

            feat = feat.cuda()
            feat = feat / feat.norm(dim = -1, keepdim = True)
            res = feat @ emb.t() / temperature

            # res = model(feat.to('cuda')) / temperature
            # print(res.shape)
            acc1 += accuracy1(res.cpu(), label)
            acc5 += accuracy5(res.cpu(), label)
            
            weight = 1 / torch.Tensor(freq).cuda()
            if mode == 'hinge': # hinge loss
                res = res[:,:866]
                loss = F.cross_entropy(res[label.cuda()!=866], label.cuda()[label.cuda()!=866], weight[:866], reduction = "sum")

                logit = F.softmax(res, dim = -1)
                bg_loss = (logit - (1/866)).maximum(torch.tensor(0).cuda()).sum(dim = -1)
                bg_loss = bg_loss[label.cuda()==866].sum() * weight[866]
                loss += bg_loss
                loss /= len(label)
                # loss /= weight[label].sum()
            
            if mode == 'mean': # mean loss
                res = res[:,:866]
                res = torch.cat([res, res.mean(dim=-1, keepdim=True)], dim = -1)
                loss = F.cross_entropy(res, label.cuda(), weight)

            if mode == 'meanbg': # mean loss only on bg
                res = res[:,:866]
                fg = label.cuda() < 866
                loss = F.cross_entropy(res[fg], label.cuda()[fg], weight[:866], reduction="sum")
                res = torch.cat([res, res.mean(dim=-1, keepdim=True)], dim = -1)
                bg = label.cuda() == 866
                loss_bg = F.cross_entropy(res[bg], label.cuda()[bg], weight, reduction="sum")
                loss = (loss + loss_bg) / len(label)

            if mode == 'max': # max loss
                res = res[:,:866]
                alpha = 1.
                res = torch.cat([res, 1-alpha*res.max(dim=-1,keepdim=True)[0]], dim = -1)
                res = F.softmax(res, dim=-1)
                loss = F.nll_loss(res.log(), label.cuda(), weight)
            
            if mode == 'learn': # learn a bg embedding
                loss = F.cross_entropy(res, label.cuda(), weight)
                # loss *= weight[0]
                # loss2 = F.cross_entropy(res, label.cuda(), reduction="mean")
                # print(weight)
                # print(loss2/loss)
            if mode == 'fg_only': # learn a bg embedding
                loss = F.cross_entropy(res, label.cuda(), weight[:866])
                # loss /= weight[label].sum()
            
            if mode == 'soft': # soft label with 1/C
                # soft_label = softLabel(label.cuda(), iou.cuda())
                res = res[:,:866]
                fg = label.cuda() < 866
                loss_fg = F.cross_entropy(res[fg], label.cuda()[fg], weight[:866], reduction="sum")
                
                bg = label.cuda() == 866
                soft_label = res.new_ones(res.shape).cuda() / 866
                loss_bg = softCrossEntropy(res[bg], soft_label[bg]) * weight[866]
                # loss_bg = F.cross_entropy(res[bg], label.cuda()[bg], weight, reduction="sum")
                loss = (loss_fg + loss_bg) / weight.cuda()[label.cuda()].sum()
                # loss *= weight[0]
                # coef = 
                # loss = (coef * loss).mean()
            
            if idb % 100 == 0:
                print("loss =", loss, f"  time={time.time()-time_s}", end='\r')
                time_s=time.time()
            model.zero_grad()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc1 = acc1.item() / len(ds.dataset)
        acc5 = acc5.item() / len(ds.dataset)
        print(f"train acc: top1={acc1}   top5={acc5}   total={len(ds.dataset)}")



# def train(model, cfg, train_set, val_set, CLASS_NAMES_FULL, base_label_ids, CLASS_THR):
#     torch.random.manual_seed(1)
#     random.seed(1)
#     RES_DIR = cfg['name']
#     MAX_EPOCH = cfg['epoch']
    
#     optimizer = SGD(model.prompt_learner.parameters(), lr=cfg['lr'])
#     lr_scheduler = build_lr_scheduler(optimizer, MAX_EPOCH, 0, 1e-5)

#     with torch.no_grad():
#         embedding = model.get_embedding()
#     test_embedding(embedding, train_set)
#     test_embedding(embedding, val_set)
#     checkpoint(model, RES_DIR+"/epoch_0")

#     for i in range(MAX_EPOCH):
#         print(f"epoch : {i+1}")
#         train_epoch(model, optimizer, train_set)
#         with torch.no_grad():
#             embedding = model.get_embedding()
#         test_embedding(embedding, train_set)
#         test_embedding(embedding, val_set)
#         checkpoint(model, f"{RES_DIR}/epoch_{i+1}")
#         lr_scheduler.step()