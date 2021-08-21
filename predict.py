import glob
import numpy as np
import torch
import os
import cv2
from unet_model import Unet


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Unet(n_channels=3, n_classes=3)
    net.to(device=device)
    net.load_state_dict(torch.load('best_model3.pth', map_location=device))
    net.eval()
    tests_path = glob.glob("test_raw/*.jpg")
    for test_path in tests_path:
        save_res_path = test_path.split('.')[0] + '_res.png'
        img = cv2.imread(test_path)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        pred = net(img_tensor)
        logit = np.array(pred.data.cpu()[0])
        pred = logit.argmax(axis=0)
        vis_base = np.zeros((img.shape[2], img.shape[3], 3), dtype=np.uint8)
        vis_base[..., 1][pred == 1] = 255
        vis_base[..., 2][pred == 2] = 255
        # pred[pred >= 0.5] = 255
        # pred[pred < 0.5] = 0
        cv2.imwrite(save_res_path, pred)
