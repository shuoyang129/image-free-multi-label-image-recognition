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

import clip
import pickle


class NUSWIDE_ZSL(data.Dataset):
    def __init__(self,
                 root,
                 data_split,
                 img_size=224,
                 p=1,
                 annFile="",
                 label_mask=None,
                 partial=1 + 1e-6):
        ann_file_names = {'train': 'formatted_train_all_labels_filtered.npy',
                          'val': 'formatted_val_all_labels_filtered.npy',
                          'val_gzsl': 'formatted_val_gzsl_labels_filtered_small.npy',
                          'test_gzsl': 'formatted_val_gzsl_labels_filtered.npy'}
        img_list_name = {'train': 'formatted_train_images_filtered.npy',
                         'val': 'formatted_val_images_filtered.npy',
                         'val_gzsl': 'formatted_val_gzsl_images_filtered_small.npy',
                         'test_gzsl': 'formatted_val_gzsl_images_filtered.npy'}
        self.root = root
        class_name_files = os.path.join(self.root, 'annotations', 'Tag_all', 'all_labels.txt')
        with open(class_name_files) as f:
            classnames = f.readlines()
        self.classnames = [a.strip() for a in classnames]

        self.classids = list(range(len(self.classnames)))
        self.name2id = dict()
        for name, id in zip(self.classnames, self.classids):
            self.name2id[name] = id

        self.data_split = data_split

        if self.data_split == 'train':
            self.train_tokenized_texts = self.read_texts_from_file(
                os.path.join(self.root, 'merge_file.txt'))    
        else:
            self.img_size = img_size

            annFile = os.path.join(self.root, 'annotations', 'zsl', ann_file_names[data_split])
            
            cls_id = pickle.load(open(os.path.join(self.root, 'annotations', 'zsl', "cls_id.pickle"), "rb"))
            cls_id = cls_id['unseen']
            self.cls_id = cls_id

            self.image_list = []
            test_image_file = os.path.join(self.root, "TestImagelist.txt")
            with open(test_image_file, 'r') as f:
                lines = f.readlines()
                for line in lines:  
                    self.image_list.append(line.strip().replace("\\", "/"))
            self.image_list = np.array(self.image_list)         
            
            self.anns = [[0] * 81 for _ in range(107859)]
            for i in range(len(self.classnames)):
                name = self.classnames[i]
                ann_file = os.path.join(self.root, "TrainTestLabels/Labels_{}_Test.txt".format(name))
                with open(ann_file, 'r') as f:
                    lines = f.readlines()
                    for j in range(len(lines)):
                        if lines[j].strip() == "1":
                            self.anns[j][i] = 1  
            self.anns = np.array(self.anns)

            assert len(self.anns) == len(self.image_list)

            self.ids = list(range(len(self.image_list)))

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
        img_id = self.ids[index]
        img_path = os.path.join(self.root, 'images', self.image_list[img_id].strip())
        img = Image.open(img_path).convert('RGB')
        targets = self.anns[img_id]
        targets = torch.from_numpy(targets).long()
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
            return len(self.ids)

    def name(self):
        return 'nus_wide_zsl'

