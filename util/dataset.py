from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os


class SpermDataset(Dataset):
    def __init__(self, src, image_size, transform=None):
        # the src directory must contain 2 folders "images" and "labels"
        # which contain identical numbers of entries.

        self.src = src
        self.image_size = image_size
        self.transform_basic = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                   ])
        self.transform = transform
        self.images = []
        self.labels = []
        self.combined = []

        for filename in os.listdir(self.src + "/images"):
            self.images.append(filename)
        for filename in os.listdir(self.src + "/labels"):
            self.labels.append(filename)

        assert len(self.images) == len(self.labels)
        for i in range(len(self.images)):
            self.combined.append((self.images[i], self.labels[i]))

    def __getitem__(self, item):
        img_name, label_name = self.combined[item]
        img = cv2.imread(self.src + "/images/" + img_name)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        label = cv2.imread(self.src + "/labels/" + label_name, 0)/255
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.combined)
