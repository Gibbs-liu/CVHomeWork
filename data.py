import torch
import cv2
import os
import glob
from torch.utils import data
from torch.utils.data import Dataset
import random
import numpy as np


class Kuang_data(Dataset):
    def __init__(self, data_path):
        super(Kuang_data, self).__init__()
        self.data_path = os.path.abspath(data_path)
        self.image_path = glob.glob(os.path.join(self.data_path,
                                                 "raw_data/*.jpg"))
        self.label_path = glob.glob(os.path.join(self.data_path,
                                                 "groundtruth/*.png"))

    def augment(self, image, flipCode):
        # 使用cv2进行数据增强，
        # 参数为1，水平翻转，参数为0，垂直翻转，-1，水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def MaxMinNormalization(self, x, Max, Min):
        x = (x-Min) / (Max - Min)
        return x

    def __getitem__(self, index):
        image_file = self.image_path[index]
        # AT： 数据要对齐
        label_file = image_file.replace("jpg", "png").replace(
            "raw_data", "groundtruth"
        )
        image = cv2.imread(image_file)
        label = cv2.imread(label_file, flags=0)
        image = image[:, :852]
        label = label[:, :852]
        # 转换为单通道
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # image = image.reshape(1, image.shape[0], image.shape[1])
        # label = label.reshape(1, label.shape[0], label.shape[1])
        # if label.max() > 1:
        #     label = label / 255
        label[label == 38] = 1
        label[label == 75] = 2
        # 随机进行数据增强
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        # image = image[np.newaxis, :, :]
        # label = label[np.newaxis, :, :]
        image = self.MaxMinNormalization(image, np.max(image), np.min(image))
        # label = self.MaxMinNormalization(label, np.max(label), np.min(label))
        # label of crossentropy loss need to be long type tensor
        return image.transpose(2, 0, 1), label.astype(np.int64)

    def __len__(self):
        return len(self.label_path)


if __name__ == "__main__":
    data = Kuang_data(os.path.abspath(os.path.join(
        os.path.dirname(__file__)
    )))
    print(os.path.abspath(__file__))
    print("all: ", len(data))
    train_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=2,
        shuffle=True
    )
    for image, label in train_loader:
        print(image.shape)
        print(label.shape)
        print(label.dtype)


