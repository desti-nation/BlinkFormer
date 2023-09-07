import json
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import print_on_rank_zero

class SynthBlinkDataset(Dataset):

    def __init__(self,
                 root_path="/home/lb/Data/04-SynthBlink-50K",
                 npy_path="/home/lb/Data/04-SynthBlink-50K/proceed_npy_50K",
                 mode="train",
                 seq_length=13,
                 transform=None,
                 eye_size=(48, 48),
                 config=None):

        self.config = config

        if mode == "train":
            with open('{}/train_data.json'.format(root_path), 'r') as fp:
                self.labels_dict = json.load(fp)
            self.folders_list = list(self.labels_dict.keys())
        elif mode == "val":
            with open('{}/val_data.json'.format(root_path), 'r') as fp:
                self.labels_dict = json.load(fp)
            self.folders_list = list(self.labels_dict.keys())
        elif mode == "test":
            with open('{}/test_data.json'.format(root_path), 'r') as fp:
                self.labels_dict = json.load(fp)
            self.folders_list = list(self.labels_dict.keys())
        else:
            raise ValueError("mode must be train, val or test !")

        self.folders_list = [(x, y) for x in self.folders_list for y in ["left", "right"]]

        self.mode = mode
        self.root_path = root_path
        self.npy_path = npy_path
        self.seq_length = seq_length
        self.transform = transform
        self.eye_size = eye_size

        print_on_rank_zero("train data:" if mode == "train" else "val data:" if mode == "val" else "test data:")
        print_on_rank_zero(len(self.folders_list))

        assert self.seq_length in [10, 13], "seq_length must be 10 or 13 !"

    def __len__(self):
        return len(self.folders_list)

    def __getitem__(self, index):

        img_folder, left_or_right = self.folders_list[index]
        eye_frame = np.load(os.path.join(self.npy_path, '{}_{}.npy'.format(img_folder, left_or_right)))
        
        with open('{}/data/{}/annotations.json'.format(self.root_path, img_folder), 'r') as fp:
            eye_annotations = json.load(fp)
        label = self.labels_dict[img_folder]  # blnik label

        eye_frame = torch.from_numpy(eye_frame).permute(0, 3, 1, 2)
        if self.transform is not None:
            eye_frame = self.transform(eye_frame)
        blink_strengths = []
        for i in range(1, 13 + 1):
            blink_strengths.append(eye_annotations[str(i).zfill(2)]['Blink_Strength'])
        blink_strengths = np.array(blink_strengths).astype(float)

        return eye_frame, label, [], img_folder, blink_strengths
