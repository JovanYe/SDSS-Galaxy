import os
import h5py
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

class GalaxyDataset():
    def __init__(self):
        with h5py.File('./data/Galaxy10.h5', 'r') as F:
            self.images = np.array(F['images'])
            self.labels = np.array(F['ans'])

        self.data_augmentation_transforms = [ # 定义基本的数据增广变换
        transforms.RandomRotation(45),  # 随机旋转
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.GaussianBlur(3, sigma=(0.1, 2)), # 随机加噪
        transforms.RandomResizedCrop(size=(69, 69), scale=(0.8, 1.2)),  # 随机缩放
        ]
        self.classes = [
            'Disk, Face-on, No Spiral',
            'Smooth, Completely round',
            'Smooth, in-between round',
            'Smooth, Cigar shaped',
            'Disk, Edge-on, Rounded Bulge',
            'Disk, Edge-on, Boxy Bulge',
            'Disk, Edge-on, No Bulge',
            'Disk, Face-on, Tight Spiral',
            'Disk, Face-on, Medium Spiral',
            'Disk, Face-on, Loose Spiral']

    def imageList(self, is_aug=False, min_size=512):
        imageList = [[], [], [], [], [], [], [], [], [], []]

        for image, label in zip(self.images, self.labels):
            imageList[label].append(torch.tensor(image).permute(2, 0, 1))

        if is_aug:
            aug_ind = range(len(self.data_augmentation_transforms))
            for i in range(len(imageList)):
                diff = min_size - len(imageList[i])
                if diff > 0:
                    sublist = imageList[i].copy()
                    sub_ind = range(len(sublist))
                    for k in range(diff):
                        transform = self.data_augmentation_transforms[random.choice(aug_ind)]
                        img = transform(sublist[random.choice(sub_ind)])
                        imageList[i].append(img)
                    #
                    # img_path = './img/' + self.classes[i]
                    # os.makedirs(img_path)
                    # for i, img in enumerate(imageList[i]):
                    #     plt.imshow(img)
                    #     plt.savefig(img_path + '/' + str(i))
                    #     plt.close()
        return imageList

    def plot(self):
        fig = plt.figure(figsize=(20, 20))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(self.images[i])
            plt.title(self.classes[self.labels[i]])
            fig.tight_layout(pad=5.0)
        plt.show()
#


if __name__ == "__main__":

    # dataset = GalaxyDataset()
    # imageList = dataset.imageList(is_aug=True, min_size=512)
    # for i in imageList:
    #     print(len(i))
    pass