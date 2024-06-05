import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import json
import jsonlines
import torchvision.transforms as transforms

from dataloaders.helper import CutoutPIL
from randaugment import RandAugment
import xml.dom.minidom
import clip
import pickle


class voc2007(data.Dataset):
    def __init__(self,
                 root,
                 data_split,
                 img_size=None,
                 p=1,
                 annFile="",
                 label_mask=None,
                 partial=1 + 1e-6):
        self.root = root
        # self.classnames = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        #                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        #                    'train', 'tvmonitor']

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

        if self.data_split == 'trainval':
            self.train_tokenized_texts = self.read_texts_from_file(
                os.path.join(self.root, 'glm_coco.txt'))
        else:
            self.img_size = img_size
            self.annFile = os.path.join(self.root, 'Annotations')
            image_list_file = os.path.join(self.root, 'ImageSets', 'Main', '%s.txt' % data_split)

            with open(image_list_file) as f:
                image_list = f.readlines()
            self.image_list = [a.strip() for a in image_list]

            self.transform = transforms.Compose([
                # transforms.CenterCrop(img_size),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

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
                classname = text[i+1].strip().lower()
                if classname in self.classnames:
                    class_idx = self.name2id[classname]
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
        img_path = os.path.join(self.root, 'JPEGImages', self.image_list[index] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        ann_path = os.path.join(self.annFile, self.image_list[index] + '.xml')
        label_vector = torch.zeros(80)
        DOMTree = xml.dom.minidom.parse(ann_path)
        root = DOMTree.documentElement
        objects = root.getElementsByTagName('object')
        for obj in objects:
            if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                continue
            tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
            label_vector[self.classnames.index(tag)] = 1.0
        targets = label_vector.long()
        target = targets[None, ]
        if self.mask is not None:
            masked = - torch.ones((1, len(self.classnames)), dtype=torch.long)
            target = self.mask[index] * target + (1 - self.mask[index]) * masked

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
            return len(self.image_list)

    def name(self):
        return 'voc2007'
