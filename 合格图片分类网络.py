import torch
import torch.nn as nn
import cv2 as cv
import os
from PIL import Image
from torchvision import transforms

'''28 train mean loss 0.0702 test mean loss 0.0444 train mean acc 0.9756 test mean acc 0.9918'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(64, 128, 1, 1, 0),
                                    nn.AdaptiveMaxPool2d(7),
                                    nn.LeakyReLU(),
                                    nn.Flatten(),
                                    nn.Linear(7 * 7 * 128, 1024),
                                    nn.Dropout(0.5),
                                    nn.Linear(1024, 2)
                                    )

    def forward(self, x):
        return self.layer1(x)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        # transforms.Resize(size=(224, 224), antialias=True)
    ])
    '''pth:./weight.pth'''
    path = './models1/weight_channel1.pth'
    # net = Net().to('cuda')
    net = Net()
    net.load_state_dict(torch.load(path))
    a, b = [], []
    for i in os.listdir('./1'):
        img = Image.open(os.path.join('./1', i))
        a.append(int(i.strip('.jpg').split('_')[-1]))
        x = transform(img)
        x = x.unsqueeze(0)
        b.append(net(x).argmax())
    print(b)
