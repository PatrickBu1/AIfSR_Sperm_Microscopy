import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import cv2
import random
import os
from typing import Union, List, Dict, Any, cast
from util.dataset import SpermDataset
from models.v_unet import VUnet
from util.loss import DiceLoss


batch_size = 32
epochs = 100
image_size = 512
train_path = r"C:/Users/jnynt/Desktop/AifSR/data_claudia/augmented_v1_train/train"
val_path = r"C:/Users/jnynt/Desktop/AifSR/data_claudia/augmented_v1_train/val"
checkpoint_path = r"C:/Users/jnynt/Desktop/AifSR/model_pth/checkpoints"

transforms = None

train_dataset = SpermDataset(train_path, image_size, transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = SpermDataset(val_path, image_size, transforms)
val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.__len__(), shuffle=True)

image_datasets = dict()
image_datasets['train'] = train_dataset
image_datasets['val'] = val_dataset
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['val'] = val_dataloader

device = torch.device('cpu')
net = VUnet(pretrained_path=r'C:/Users/jnynt/Desktop/AifSR/model_pth/vunet_parsed.pth')
# net.to(device)
net.freeze() # freezes the down sampling path of the network

optimizer = optim.Adam(net.parameters(), lr=0.00001)
loss = DiceLoss()





