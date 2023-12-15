import os

import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
from train import Net

np.set_printoptions(precision=4, suppress=True)


def softmax(z):
    # 计算softmax函数
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


if __name__ == '__main__':
    ts = transforms.Compose([transforms.Grayscale(),
                             transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
                             transforms.ToTensor()])

    model = Net()
    state_dict = torch.load("./models/model_0010.pth")
    model.load_state_dict(state_dict)
    model.eval()

    # model = torch.jit.load("./models/scriptmodel.pth")
    # model.eval()

    with torch.no_grad():
        # 单张图片
        # img = Image.open(r"test_images/qualifications/09c595e91e61688bafdb6f8e1dcac638.jpg").convert('RGB')
        # img = ts(img)
        # img = img[None, ...]
        # r = model(img)
        # print(r)  # (1,2)
        # print(softmax(r[0].numpy()))
        # print(torch.argmax(r).item())
        # 多张图片
        src = "./test_images/qualifications"  # 合格图片
        # src = "./test_images/disqualification"  # 不合格图片
        for i in os.listdir(src):
            img = Image.open(os.path.join(src, i))
            img = ts(img)
            img = img[None, ...]
            r = model(img)
            # print(r)  # (1,2)
            # print(softmax(r[0].numpy()))
            print(torch.argmax(r).item())
