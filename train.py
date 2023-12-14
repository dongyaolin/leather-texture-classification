import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torchvision.models import VGG16_BN_Weights, ResNet50_Weights, GoogLeNet_Weights

import numpy as np
import os

from dataset_v2 import DataSet_V2
from metrics import AccuracyScore

torch.set_printoptions(precision=2, sci_mode=False)


class DogCatClassifier_V5:
    def __init__(self, model, train_data_dir, test_data_dir):
        self.batch_size = 64
        self.num_workers = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.total_epoch = 100
        self.lr = 0.005
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = AccuracyScore()
        self.opt = optim.SGD(
            params=[p for p in self.model.parameters() if p.requires_grad is True],
            lr=self.lr
        )
        self.print_interval = 2
        self.model_dir = 'models'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            # names = os.listdir(self.model_dir)
            # if len(names) > 0:
            #     names.sort()
            #     name = names[-1]
            #     missing_keys, unexpected_keys = self.model.load_state_dict(
            #         torch.load(os.path.join(self.model_dir, name)))
            ...
        self.model = self.model.to(self.device)  # 注意这一行要放在后面

    def save_model(self, epoch):
        # 模型保存
        if epoch == self.total_epoch:
            model_path = os.path.join(self.model_dir, "best.pth")
        else:
            model_path = os.path.join(self.model_dir, f"model_{epoch:04d}.pth")
        torch.save(self.model.state_dict(), model_path)

    def train(self):
        # 1. 加载数据
        trainset = DataSet_V2(root_dir=self.train_data_dir,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              istrainning=True)
        testset = DataSet_V2(root_dir=self.test_data_dir,
                             batch_size=self.batch_size,
                             shuffle=False,
                             num_workers=self.num_workers,
                             istrainning=False)

        for epoch in range(self.total_epoch):
            self.model.train(True)  # Sets the module in training mode.
            train_loss = []
            train_acc = []
            batch = 0
            for inputs, labels in trainset:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)

                # backward
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                acc = self.acc_fn(output, labels)

                train_loss.append(loss.item())
                train_acc.append(acc)
                if batch % self.print_interval == 0:
                    print(f'{epoch + 1}/{self.total_epoch} {batch} train_loss={loss.item()} -- {acc.item():.4f}')
                batch += 1

            test_loss = []
            test_acc = []
            batch = 0
            for data in testset:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)
                acc = self.acc_fn(output, labels)

                test_loss.append(loss.item())
                test_acc.append(acc.item())
                if batch % self.print_interval == 0:
                    print(f'{epoch + 1}/{self.total_epoch} {batch} test_loss={loss.item()} -- {acc.item():.4f}')
                batch += 1
                self.save_model(epoch)

            print(f'{epoch} train mean loss {np.mean(train_loss):.4f} test mean loss {np.mean(test_loss):.4f}'
                  f' train mean acc {np.mean(train_acc):.4f} test mean acc {np.mean(test_acc):.4f}')
            self.save_model(self.total_epoch)


if __name__ == '__main__':
    # vgg = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    # googlenet = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    # googlenet = models.googlenet()
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # vgg.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
    # googlenet.fc = nn.Linear(in_features=1024, out_features=2, bias=True)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.fc = nn.Linear(in_features=2048, out_features=27, bias=True)
    net = Net()
    train_data_dir = './identify_dataset/training_data'
    test_data_dir = './identify_dataset/testing_data'
    # print(resnet.conv1)
    model = DogCatClassifier_V5(resnet, train_data_dir, test_data_dir)
    model.train()
