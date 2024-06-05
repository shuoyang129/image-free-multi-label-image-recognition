import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import json
import jsonlines
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import random

import clip
import pickle


class Coco(data.Dataset):
    def __init__(self,
                 root,
                 data_split,
                 img_size=224,
                 p=1,
                 annFile="",
                 label_mask=None,
                 partial=1 + 1e-6):
        self.root = root
        self.classnames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                           "kite",
                           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                           "orange",
                           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
                           "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]

        self.classids = list(range(len(self.classnames)))
        self.name2id = dict()
        for name, id in zip(self.classnames, self.classids):
            self.name2id[name] = id

        self.data_split = data_split

        if self.data_split == 'train2014':
            self.train_tokenized_texts = self.read_texts_from_file(
                os.path.join(self.root, 'glm_coco.txt'))    
        else:
            self.img_size = img_size

            if annFile == "":
                # annFile = os.path.join(self.root, 'annotations', 'instances_%s.json' % data_split)
                annFile = os.path.join(self.root, 'annotations', 'instances_%s.json' % data_split)
                cls_id = list(range(len(self.classnames)))
            else:
                cls_id = pickle.load(open(os.path.join(self.root, 'annotations', "cls_ids.pickle"), "rb"))
                if "unseen" in annFile:
                    cls_id = cls_id["test"]
                else:
                    cls_id = cls_id['train'] | cls_id['test']
                cls_id = list(cls_id)
            cls_id.sort()
            self.coco = COCO(annFile)
            ids = list(self.coco.imgToAnns.keys())
            self.ids = ids

            self.transform = transforms.Compose([
                # transforms.CenterCrop(img_size),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

            self.cat2cat = dict()
            cats_keys = [*self.coco.cats.keys()]
            cats_keys.sort()
            for cat, cat2 in zip(cats_keys, cls_id):
                self.cat2cat[cat] = cat2
            self.cls_id = cls_id

            # create the label mask
            self.mask = None
            self.partial = partial

    def read_texts_from_file(self, file_path):
        lines = open(file_path, 'r').readlines()
        train_tokenized_texts = []
        for line in lines:
            text = line.split('#####')
            try:
                tokenized_texts = clip.tokenize(text[0].strip())
            except:
                continue

            label = [0] * len(self.classnames)
            for i in range(len(text)-1):
                class_idx = self.name2id[text[i+1].strip().lower()]
                label[int(class_idx)] = 1
            train_tokenized_texts.append((tokenized_texts, label))
        print("length of training text: ", len(train_tokenized_texts))
        return train_tokenized_texts
    
    def read_texts_from_json(self, file_path):
        with open(file_path, 'r') as f:
            list = json.load(f)
        train_tokenized_texts = []
        for text, label in list:
            try:
                tokenized_texts = clip.tokenize(text.strip())
            except:
                continue
            train_tokenized_texts.append((tokenized_texts, label))
        print("length of training text: ", len(train_tokenized_texts))
        return train_tokenized_texts

    def get_train_data(self, index):
        tokenized_texts, target = self.train_tokenized_texts[index]
        target = torch.tensor(target).long()
        target = target[None, :]
        return tokenized_texts[0], target

    def get_test_data(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, len(self.classnames)), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output
        if self.mask is not None:
            masked = - torch.ones((3, len(self.classnames)), dtype=torch.long)
            target = self.mask[index] * target + (1 - self.mask[index]) * masked

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, self.data_split, path)).convert('RGB')
    
        if self.transform is not None:
            img = self.transform(img)
    
        return img, target

    def __getitem__(self, index):
        if 'train' in self.data_split:
            return self.get_train_data(index)
        else:
            return self.get_test_data(index)

    def __len__(self):
        if 'train' in self.data_split:
            return len(self.train_tokenized_texts)
        else:
            return len(self.ids)

    def name(self):
        return 'coco'
