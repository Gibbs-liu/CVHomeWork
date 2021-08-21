# -*- encoding: utf-8 -*-
'''
@File    :      train.py
@Time    :      2021/08/20 21:26:00
@Author  :      liu haoyu
@Version :      1.0
@Contact :      gibbsliuhy@gmail.com
@License :      (C)Copyright 2020-2021, NLPR-CASIA
@Desc    :      训练数据
'''

# here put the import lib
from cv2 import preCornerDetect
from torch._C import dtype
from unet_model import Unet
from data import Kuang_data
from torch import optim
import torch.nn as nn
import torch
import os
from losses import *


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.01):
    # 加载数据
    kuang_dataset = Kuang_data(os.path.abspath(
        os.path.dirname(__file__)
    ))
    train_loader = torch.utils.data.DataLoader(
        dataset=kuang_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    # RMSprop
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8,
                              momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # criterion = nn.BCEWithLogitsLoss()  # 这个不行，要换
    # criterion = dice_loss
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        net.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device) # 要转tensor
            pred = net(image)
            loss = criterion(pred, label)
            print(f'Loss/train {epoch}/{epochs}', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model3.pth')
            loss.backward()
            optimizer.step()
            
            
if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net = Unet(n_channels=3, n_classes=3)
    net.to(device=device)
    data_path = '.'
    train_net(net, device, data_path, epochs=50, batch_size=1)




