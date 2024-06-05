import os
import sys
sys.path.insert(0, '../')
import torch
import time
import json
from tqdm import tqdm

from utils.helper import AverageMeter, mAP, calc_F1
from torch.cuda.amp import autocast
import numpy as np
from ipdb import set_trace
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"
zs_clip, preprocess = clip.load("RN50", device=device)

def validate_baseline(data_loader,
                      model,
                      args,
                      datasetname,
                      GL_merge_rate=0.5,
                      logdir=None):
    batch_time = AverageMeter()

    model.eval()

    preds = []
    targets = []
    output_idxs = []
    
    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in tqdm(enumerate(data_loader)):
            target = target.max(dim=1)[0]
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            inputs = inputs.to(device)
            shape = inputs.shape

            # compute output
            with autocast():
                output, output_local,_, _, _ = model(inputs, if_test=True)
            length = output.shape[-1]

            tmp = 0.65
            pred_merge = output * tmp + output_local * (1 - tmp)

            preds.append(pred_merge.cpu())
            targets.append(target.cpu())
            output_idx = pred_merge.argsort(dim=-1, descending=True)
            output_idxs.append(output_idx)

            batch_time.update(time.time() - end)
            end = time.time()

        mAP_score,ap = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy(), datasetname, False)

        precision_3, recall_3, F1_3 = calc_F1(torch.cat(targets, dim=0).cpu().numpy(), torch.cat(output_idxs, dim=0).cpu().numpy(), args.top_k,
                                            num_classes=length)
        # print(mAP_score)
        # print(precision_3)
        # print(recall_3)
        # print(F1_3)
        # print(ap)
    torch.cuda.empty_cache()
    return F1_3, mAP_score
