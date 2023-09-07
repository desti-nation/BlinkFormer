import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader

from HUSTDataset import HUSTDataset
import data_transform as T
from SynthBlink_50K_Dataset_npy import SynthBlinkDataset
from torchvision import transforms


class HUSTDataModule(pl.LightningDataModule):
	def __init__(self, configs):
		super().__init__()
		self.configs = configs

	def get_dataset(self, isTrain, transform):
		dataset = HUSTDataset(root_path="/home/lb/Data/01-Blink--HUST-LEBW", isTrain = isTrain, seq_length = 13, eye_size = (48, 48), transform=transform, config=self.configs)
		return dataset

	def setup(self, stage):
		
		mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
		
		train_transform = T.Compose([
				transforms.RandomHorizontalFlip(p=0.5),
				transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,2.0),saturation=(0.5,1.5),hue=(-0.5,0.5)),
				transforms.RandomGrayscale(0.25),
				transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
				T.ToTensor(),
				T.Normalize(mean, std),
				transforms.RandomApply([transforms.ColorJitter(brightness=(0.2, 0.8))], p=0.3),
				])

		self.train_dataset = self.get_dataset(True, train_transform)

		test_transform = T.Compose([
				T.ToTensor(),
				T.Normalize(mean, std),
				])

		self.test_dataset = self.get_dataset(False, test_transform)
		self.val_dataset = self.test_dataset

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.configs.batch_size,
			num_workers=self.configs.num_workers,
			shuffle=True,
			drop_last=True, 
			pin_memory=False
		)
	
	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.configs.batch_size,
			num_workers=self.configs.num_workers,
			shuffle=False,
			drop_last=False,
		)
	
	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.configs.batch_size,
			num_workers=self.configs.num_workers,
			shuffle=False,
			drop_last=False,
		)

class SynthBlinkNPYDataModule(pl.LightningDataModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    def get_dataset(self, mode, transform):
        dataset = SynthBlinkDataset(mode=mode, transform=transform, config=self.configs)
        return dataset

    def setup(self, stage):

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

		# from hust best experience
        train_transform = T.Compose([
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,2.0),saturation=(0.5,1.5),hue=(-0.5,0.5)),
			transforms.RandomGrayscale(0.25),
			transforms.RandomApply([transforms.RandomRotation(90)], p=0.3),
			# transforms.RandomRotation(10), # -10 ~ +10
			# FrameReversal(p=0.5),
			# transforms.RandomApply(frame_argus, p=0.1),
			T.ToTensor(),
			T.Normalize(mean, std),
			transforms.RandomApply([transforms.ColorJitter(brightness=(0.2, 0.8))], p=0.3),
			])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        val_transform = test_transform

        self.train_dataset = self.get_dataset("train", train_transform)
        self.val_dataset = self.get_dataset("val", val_transform)
        self.test_dataset = self.get_dataset("test", test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.configs.batch_size,
            num_workers=self.configs.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.configs.batch_size,
            num_workers=self.configs.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.configs.batch_size,
            num_workers=self.configs.num_workers,
            shuffle=False,
            drop_last=False,
        )


if __name__ == "__main__":
	pass