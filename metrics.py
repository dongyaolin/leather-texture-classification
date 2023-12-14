import torch.nn as nn
from torch import Tensor
import torch


class AccuracyScore(nn.Module):
    def __init__(self):
        super(AccuracyScore, self).__init__()

    def forward(self, input: Tensor, target: Tensor):
        """
        计算准确率
        :param input: [N,C]N表示样本数目，C表示类别数目。也就是存在N个样本，每个样本属于C个类别的置信度
        :param target:  [N,]N表示样本数目，也就是N个样本属于C个类别中的序号
        :return: 准确率
        """
        pred = torch.argmax(input, dim=1)  # 按照dim=1这个维度，提取置信度最大的值对应的序号
        # print(pred.dtype)  # 是torch.int64
        # print(target.dtype)  # 是torch.float32
        pred = pred.to(target.dtype)  # 类型转换
        correct = pred == target
        acc = torch.mean(correct.to(torch.float32))
        return acc.cpu()


if __name__ == '__main__':
    input = torch.Tensor(
        [[0.1, 0.9], [0.8, 0.2], [0.8, 0.2], [0.7, 0.3]]
    )
    target = torch.Tensor(
        [1, 0, 0, 1]
    )
    acc = AccuracyScore()
    acc = acc(input, target)
    print(acc)
