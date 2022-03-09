import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from typing import Union, List, Dict, Any, cast
from util.dataset import SpermDataset
from models.v_unet import VUnet
from util.loss import DiceLoss, DiceBCELoss, IoULoss
from torchsummary import summary


batch_size = 8
epochs = 100
image_size = 512
train_path = r"C:/Users/jnynt/Desktop/AifSR/data_claudia/augmented_v1_train/train"
val_path = r"C:/Users/jnynt/Desktop/AifSR/data_claudia/augmented_v1_train/val"
checkpoint_path = r"C:/Users/jnynt/Desktop/AifSR/model_pth/checkpoints"

data_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.5), (0.5))])

train_dataset = SpermDataset(train_path, image_size, data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = SpermDataset(val_path, image_size, data_transform)
val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.__len__(), shuffle=True)

image_datasets = dict()
image_datasets['train'] = train_dataset
image_datasets['val'] = val_dataset
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['val'] = val_dataloader

device = torch.device('cpu')
net = VUnet(pretrained_path=r'C:/Users/jnynt/Desktop/AifSR/model_pth/vunet_parsed.pth')
summary(net, input_size=(3, 512, 512))
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1E-5)
criterion = DiceBCELoss()
acc_criterion = DiceLoss()

for epoch in range(1, epochs + 1):
    print('Epoch {}/{}'.format(epoch, epochs))
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()  # Set model to training mode
        else:
            net.eval()  # Set model to evaluate mode

        net.freeze()    # freeze up the encoder part of the network to prevent training

        running_loss = 0.0
        running_accs = 0.0

        n = 0
        for data in dataloaders[phase]:
            if n % 5 == 0:
                print('- Batch ', str(n+1))
            images, labels = data
            images, labels = images.to(device).float(), labels.to(device).float()
            output = net(images)                # get result from network
            output = torch.squeeze(output)
            loss = criterion(output, labels)    # generate DiceBCELoss loss
            output_masks = output.cpu().data.numpy().copy()
            y_mask = labels.cpu().data.numpy().copy()
            output_masks = (output_masks > 0.5)
            output = (output > 0.5)
            inter = torch.dot(output.view(-1).float(), labels.view(-1))
            union = torch.sum(output.view(-1).float()) + torch.sum(labels.view(-1)) - inter + 0.0001
            # Calculate DICE
            acc = inter / union

            optimizer.zero_grad()
            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.data.item()
            running_accs += acc
            n += 1

        epoch_loss = running_loss / n
        print(str(epoch_loss))
        epoch_acc = running_accs / n
        print(str(running_loss))

        if phase == 'train':
            print('train epoch_{} loss=' + str(epoch_loss).format(epoch))
            print('train epoch_{} dice=' + str(epoch_acc).format(epoch))
        else:
            print('val epoch_{} loss=' + str(epoch_loss).format(epoch))
            print('val epoch_{} dice=' + str(epoch_acc).format(epoch))

    if epoch % 10 == 0:
        torch.save(net, checkpoint_path + '/model_epoch_{}.pth'.format(epoch))
        print('model_epoch_{}.pth saved!'.format(epoch))



