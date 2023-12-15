import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
from torchvision.models import ResNet50_Weights, ResNet34_Weights, ResNet18_Weights
import torch.nn as nn

np.set_printoptions(precision=4, suppress=True)


def softmax(z):
    # 计算softmax函数
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


if __name__ == '__main__':
    ts = transforms.Compose([transforms.Grayscale(),
                             transforms.ToTensor()])

    model = models.resnet50()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=27, bias=True)
    '''单通道效果差些'''
    # state_dict = torch.load("./models/resnet50_channel3.pth")
    state_dict = torch.load("./models/resnet50_channel1.pth")
    model.load_state_dict(state_dict)
    model.eval()

    # model = torch.jit.load("./models/scriptmodel.pth")
    # model.eval()

    with torch.no_grad():
        img = Image.open(r".\test_images\class_1\0cc54a51a18783c00f1a3c7202df3837.jpg").convert("RGB")
        img = ts(img)
        img = img[None, ...]
        r = model(img)
        print(r[0][:6])
        print(softmax(r[0][:6].numpy()))
        print(torch.argmax(r).item())
