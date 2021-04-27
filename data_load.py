import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, base_path, transform=None):
        imgs = []
        labels = []
        img_path = os.path.join(base_path, 'images')
        label_path = os.path.join(base_path, 'masks')
        img_list = os.listdir(img_path)
        label_list = os.listdir(label_path)
        for i in range(len(img_list)):
            imgs.append(os.path.join(img_path, img_list[i]))
            labels.append(os.path.join(label_path, label_list[i]))
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        label = cv2.imread(label, 0)
        label[label > 0] = 1
        label = np.expand_dims(label, axis=0)
        label = np.array(label, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = np.array(img / 255, dtype=np.float32)
        return img, label

    def __len__(self):
        return len(self.imgs)
