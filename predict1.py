import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import torch.nn as nn

np.set_printoptions(precision=4, suppress=True)


def softmax(z):
    # 计算softmax函数
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


class MyModel:
    def __init__(self, parm="./models1/model_0350.pth"):
        self.ts = transforms.Compose([transforms.Grayscale(),
                                      transforms.ToTensor()])
        self.parm = parm
        self.model = models.resnet18().to('cpu')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=27, bias=True)
        state_dict = torch.load(self.parm, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            img = self.ts(x)
            img = img[None, ...]
            res = self.model(img)
            prediction = res[0]
            # 将类别和得分组合在一起
            class_scores = [(class_index, score) for class_index, score in enumerate(prediction)]

            # 按照得分对类别进行排序
            sorted_classes = sorted(class_scores, key=lambda x: x[1], reverse=True)

            # 获取最接近的六个类别
            top_6_classes = sorted_classes[:6]

            # 输出最接近的六个类别及其得分
            # for i, (class_index, score) in enumerate(top_6_classes):
            #     print(f"Class {class_index + 1}: Score {score}")

            class_map = {0: 'caoweihua1', 1: 'caoweihua2', 2: 'chenjunqian1', 3: 'chenjunqian2', 4: 'chenjunqian3',
                         5: 'chenjunqian4', 6: 'dyl1', 7: 'dyl2', 8: 'guoxy1', 9: 'guoxy2', 10: 'ldy1', 11: 'ldy2',
                         12: 'ldy3',
                         13: 'ldy4', 14: 'ldy5', 15: 'ldy6', 16: 'ldy7', 17: 'liuhanxin1', 18: 'liuhanxin2',
                         19: 'renzunmin1',
                         20: 'renzunmin2', 21: 'renzunmin3', 22: 'yanzhidong1', 23: 'yanzhidong2', 24: 'yuyiling1',
                         25: 'yuyiling2',
                         26: 'yuyiling3'}
            # 返回最接近的六个类别的索引
            top_6_class_indices = [class_map[class_index] for class_index, _ in top_6_classes]
            # print(top_6_class_indices)

        return top_6_class_indices


if __name__ == '__main__':
    x = Image.open(r'identify_dataset\testing_data\class_1\0cc54a51a18783c00f1a3c7202df3837.jpg')
    model = MyModel()
    print(model.predict(x))

