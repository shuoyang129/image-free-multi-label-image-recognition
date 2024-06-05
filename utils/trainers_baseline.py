import sys
sys.path.insert(0, '../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils.data_utils import make_zipfile
from utils.helper import AverageMeter, mAP
from utils.loss import ranking_loss
from torch.cuda.amp import autocast
import numpy as np
import scipy.stats
import json
    

def train_baseline(data_loader,
                   val_loaders,
                   model,
                   optim,
                   sched,
                   args,
                   cfg,
                   epoch,
                   cls_id=None):
    batch_time = AverageMeter()
    mAP_batches = AverageMeter()
    losses = AverageMeter()

    model.eval()
    if not isinstance(model, nn.DataParallel):
        model.prompt_learner.train()
        if cfg.TRAINER.FINETUNE_ATTN:
            model.model.visual.attnpool.train()

        if cfg.TRAINER.FINETUNE_BACKBONE:
            model.model.visual.train()
    else:
        model.module.prompt_learner.train()
        if cfg.TRAINER.FINETUNE_ATTN:
            model.model.module.visual.attnpool.train()

        if cfg.TRAINER.FINETUNE_BACKBONE:
            model.model.module.visual.train()

    end = time.time()
    
    if cfg.DATASET.NAME in ["coco", "voc2007"]:
        filename = './data/coco-2014/similarity_hand.json'
    elif cfg.DATASET.NAME == "nus_wide_zsl":
        filename = './data/nus_wide/similarity_hand_nus.json'
    # elif cfg.DATASET.NAME == "voc2007":
    #     filename = './data/VOCdevkit/VOC2007/similarity_hand_voc.json'
    with open(filename, 'r') as f:
        similarity_t = json.load(f)
    similarity_t = torch.tensor(similarity_t).cuda()
    
    numx = 0
    for i, (inputs, target) in enumerate(data_loader):
        numx += inputs.shape[0]
        target = target.max(dim=1)[0]
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        inputs = inputs.to(device)
        target = target.to(device)
 
        with autocast():
            output, output_local, text_features_n, text_features = model(
                None, inputs, if_test=False)

        length = output.shape[-1]
        similarity = (100*text_features@text_features.T)
        element = []
        for k in range(length):
            for j in range(k+1, length):
                element.append(similarity[k,j].unsqueeze(0))
        element=torch.cat(element)
        element = element.softmax(dim=-1)

        loss = ranking_loss(output, target, scale_=1.0,
                            margin_=1) + ranking_loss(
                                output_local, target, scale_=1.0, margin_=1)

        loss2 = F.kl_div(element.log(), 
                        similarity_t.detach(), reduction='sum')
        
        loss = loss + 0.2*loss2

        # update the network
        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.update(loss.item(), inputs.size(0))

        tmp = 0.65
        pred = output * tmp + output_local * (1 - tmp)
        
        mAP_value, _= mAP(target.cpu().numpy(), pred.detach().cpu().numpy(), cfg.DATASET.NAME, True)
        mAP_batches.update(mAP_value, inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.2f} ({losses.avg:.2f})\t'
                  'mAP {mAP_batches.val:.2f} ({mAP_batches.avg:.2f})'.format(
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      losses=losses,
                      mAP_batches=mAP_batches),
                  flush=True)

    sched.step()

    return batch_time, losses, mAP_batches