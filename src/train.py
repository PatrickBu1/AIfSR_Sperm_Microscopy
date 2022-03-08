import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import random
import os
from typing import Union, List, Dict, Any, cast


def train(epochs, batch_size, image_size, train_path, val_path):
    # TODO: write it.