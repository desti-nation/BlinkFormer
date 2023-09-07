# coding: utf-8


# ---------------------------------------
# 单只眼睛
# ---------------------------------------

from torchvision import transforms
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import glob
import json
import torch


# 单只眼睛
class HUSTDataset(Dataset):

    def __init__(self, 
        root_path="/home/lb/Data/01-Blink--HUST-LEBW", 
        isTrain = True, 
        seq_length = 13,
        eye_size = (48, 48),
        transform = None,
        config = None):

        self.config = config
        
        if isTrain:
            with open('{}/train_data.json'.format(root_path), 'r') as fp:
                self.labels_dict = json.load(fp)
            self.folders_list = list(self.labels_dict.keys())
        else:
            with open('{}/test_data.json'.format(root_path), 'r') as fp:
                self.labels_dict = json.load(fp)
            self.folders_list = list(self.labels_dict.keys())

        self.isTrain = isTrain
        self.root_path = root_path
        self.seq_length = seq_length
        self.eye_size = eye_size
        self.transform = transform

        # 输入单只眼睛的序列
        folders = []
        self.labels = []
        for folder in self.folders_list:
            label = self.labels_dict[folder]
            zuo_folder = "{}/{}/{}/zuo".format(self.root_path, folder, self.seq_length)
            you_folder = "{}/{}/{}/you".format(self.root_path, folder, self.seq_length)
            if os.path.exists(zuo_folder):
                folders.append(zuo_folder)
                self.labels.append(label)
            if os.path.exists(you_folder):
                folders.append(you_folder)
                self.labels.append(label)            

        self.folders_list = folders

        print("train data:" if isTrain else "test data:", len(self.folders_list))

        assert self.seq_length in [10, 13], "seq_length must be 10 or 13 !"

    def __len__(self):
        return len(self.folders_list)

    def __getitem__(self, index):

        # label = self.labels_dict[self.folders_list[index]]
        label = self.labels[index]

        img_folder = self.folders_list[index]

        eye_images = []
        
        eye_paths = glob.glob("{}/*.bmp".format(img_folder))
        
        if eye_paths:
            eye_paths.sort(key=lambda item: int(item[:-11].split("/")[-1])) # 1...13 排序
            for eye_path in eye_paths:
                image = cv2.imread(eye_path)
                image = cv2.resize(image, self.eye_size, interpolation=cv2.INTER_CUBIC)
                eye_images.append(image)
            eye_images = np.array(eye_images) # 自动扩维拼接 list里的image array
        else:
            eye_images = np.zeros((self.seq_length, self.eye_size[0], self.eye_size[1], 3))
            label = 0

        frames_orig = eye_images.copy()

        # Video align transform: T C H W
        with torch.no_grad():
            eye_images = torch.from_numpy(eye_images).permute(0,3,1,2)
            if self.transform is not None:
                eye_images = self.transform(eye_images)
            eye_images = eye_images.float()
            if self.config is not None and self.config.arch == 'c3d':
                eye_images = eye_images.permute(1, 0, 2, 3)
        # frames_orig = []
        img_folders = []
        annos = []
        return (eye_images, label, frames_orig, img_folders, annos)

if __name__ == '__main__':
    hust = HUSTDataset(root_path="/home/lb/Data/01-Blink--HUST-LEBW", isTrain = True, seq_length = 13, eye_size = (48, 48))
    loader = DataLoader(dataset=hust, batch_size=4, shuffle=True)
    for index, (eye_images, label, _, _, _) in enumerate(loader):
        print(eye_images.shape) # torch.Size([4, 13, 3, 48, 48])
        pass
