import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torchvision import models
from torchvision.models import VGG16_BN_Weights, ResNet50_Weights, GoogLeNet_Weights
import numpy as np
import os
from dataset import DataSet
from metrics import AccuracyScore
import pandas as pd

torch.set_printoptions(precision=2, sci_mode=False)


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


class Classifier:
    def __init__(self, model, train_data_dir, test_data_dir, fig):
        self.batch_size = 64
        self.num_workers = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.total_epoch = 100
        self.lr = 0.003
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = AccuracyScore()
        self.opt = optim.SGD(
            params=[p for p in self.model.parameters() if p.requires_grad is True],
            lr=self.lr
        )
        self.print_interval = 2
        self.model_dir = 'models'
        self.fig = True
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists("./fig"):
            os.makedirs("./fig")
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
        trainset = DataSet(root_dir=self.train_data_dir,
                           batch_size=self.batch_size,
                           shuffle=True,
                           num_workers=self.num_workers,
                           istrainning=True)
        testset = DataSet(root_dir=self.test_data_dir,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          istrainning=False)
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        for epoch in range(self.total_epoch):
            self.model.train(True)  # Sets the module in training mode.

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
                if epoch+1 % 10 == 0:
                    self.save_model(epoch)

            print(f'{epoch} train mean loss {np.mean(train_loss):.4f} test mean loss {np.mean(test_loss):.4f}'
                  f' train mean acc {np.mean(train_acc):.4f} test mean acc {np.mean(test_acc):.4f}')
        df_train = pd.DataFrame(
            {'train_loss': train_loss, 'train_acc': train_acc})
        df_train.index = range(1, len(train_loss) + 1)
        df_train.to_csv('./fig/train_log.csv')
        df_test = pd.DataFrame(
            {'test_loss': test_loss, 'test_acc': test_acc})
        df_test.index = range(1, len(test_loss) + 1)
        df_test.to_csv('./fig/test_log.csv')
        if self.fig:
            plt.figure(figsize=(12, 8), dpi=200)
            plt.plot(df_train.index, df_train.loc[:, 'train_loss'], label='train_loss')
            plt.plot(df_train.index, df_train.loc[:, 'train_acc'], label='train_acc')
            plt.plot(df_test.index, df_test.loc[:, 'test_loss'], label='test_loss')
            plt.plot(df_test.index, df_test.loc[:, 'test_acc'], label='test_acc')
            plt.legend()
            plt.title('loss&acc')
            plt.xlabel('epoch')
            plt.savefig(f'./fig/{time.strftime("%d%H%M")}.jpg')
            plt.pause(1)


if __name__ == '__main__':
    if 1:
        start = time.time()
        train_data_dir = './row_dataset/training_data'
        test_data_dir = './row_dataset/testing_data'
        net = Net()
        model = Classifier(net, train_data_dir, test_data_dir, fig=True)
        model.train()
        end = time.time()
        print(f'{net} training time:{end - start:.2f}s')

    # train
    if 0:
        start = time.time()
        train_data_dir = './identify_dataset/training_data'
        test_data_dir = './identify_dataset/testing_data'
        # vgg = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        # googlenet = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        # googlenet = models.googlenet()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # vgg.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
        # googlenet.fc = nn.Linear(in_features=1024, out_features=2, bias=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = nn.Linear(in_features=2048, out_features=27, bias=True)
        model = Classifier(resnet, train_data_dir, test_data_dir, fig=1)
        model.train()
        end = time.time()
        print(f'{model} training time:{end - start:.2f}s')
