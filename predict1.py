import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
from train import Net
from scripts.video2img import VideoFrameExtractor
import cv2

np.set_printoptions(precision=4, suppress=True)


def softmax(z):
    # 计算softmax函数
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class MyModel:
    def __init__(self, parm0='./models/model_0090.pth', parm1="./models1/model_0350.pth"):
        self.ts = transforms.Compose([transforms.Grayscale(),
                                      transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
                                      transforms.ToTensor()])
        self.parm0 = parm0

        self.model0 = Net().to('cpu')
        self.model0.load_state_dict(torch.load(self.parm0, map_location=torch.device('cpu')))
        self.parm = parm1
        self.model = models.resnet18().to('cpu')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=27, bias=True)
        self.model.load_state_dict(torch.load(self.parm, map_location=torch.device('cpu')))
        self.model.eval()

    def forward(self, x):
        return self.model(x)

    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.ts(img)
        return img

    def predict(self, video_path):

        # 初始化模型

        # 打开视频文件

        cap = cv2.VideoCapture(video_path)

        best_prediction = ''
        best_confidence = 0.0
        best_frame = None

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # 在这里将 OpenCV 的帧转换为 PIL 图像（可能需要一些格式转换）
            # 这里假设 preprocess_image 是你用来处理图像的函数
            # preprocess_image 需要根据你的模型输入进行适当的处理
            image = self.preprocess_image(frame)
            image = image.unsqueeze_(0)
            # 使用模型进行预测
            self.model0.eval()
            with torch.no_grad():
                output = self.model0(image)  # (1, 2)
                output = torch.nn.functional.softmax(output, dim=1)
                # output = dict('0':output[0][0],'1':output[0][1])
            # 记录每个预测结果的指标
            # 这里假设模型输出是一个字典，包括 'prediction' 和 'confidence' 键
            #     prediction = output['prediction']
            #     confidence = output['confidence']
            #
            #     # 判断当前预测结果是否比之前的更好
            #     if confidence > best_confidence:
            #         best_prediction = prediction
            #         best_confidence = confidence
            #         best_frame = frame  # 记录当前最好的帧图像
            #
            # # 处理最好的预测结果和图像
            # print("Best Prediction:", best_prediction)
            # 处理 best_frame（这里可以输出、保存或进一步处理图像）

            cap.release()

    def predict1(self, x):
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
    # x = Image.open(r'identify_dataset\testing_data\class_1\0cc54a51a18783c00f1a3c7202df3837.jpg')
    model = MyModel()
    # print(model.predict(x))
    model.predict(r'./exam.mp4')
