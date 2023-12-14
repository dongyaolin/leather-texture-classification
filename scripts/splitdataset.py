"""该脚本对原始已经做好类别分类的数据集进行训练集和测试集的划分"""


import os
import shutil
import math


def split_data(src_path, train_path, test_path, ratio):
    # 创建训练集标签文件夹
    dir_list = os.listdir(src_path)
    for dir in dir_list:
        if not os.path.exists(os.path.join(train_path, dir)):
            os.mkdir(os.path.join(train_path, dir))
        else:
            # print(f'{os.path.join(train_path, dir)}已存在')
            pass
    # 创建测试集标签文件夹
    for dir in dir_list:
        if not os.path.exists(os.path.join(test_path, dir)):
            os.mkdir(os.path.join(test_path, dir))
        else:
            # print(f'{os.path.join(test_path, dir)}已存在')
            pass
    # 划分数据集
    for dir in os.listdir(src_path):
        src = os.path.join(src_path, dir)
        current_len = len(os.listdir(src))
        test_count = round(current_len * ratio)
        # 测试集
        # print(type(os.listdir(src)))
        for i in os.listdir(src)[0:test_count]:
            file = os.path.join(src, i)
            if not os.path.exists(test_path + '/' + file.split('\\')[-2] + '/' + i):
                shutil.move(file, test_path + '/' + file.split('\\')[-2] + '/' + i)
            else:
                pass
        # 训练集
        for i in os.listdir(src)[test_count:current_len]:
            file = os.path.join(src, i)
            if not os.path.exists(train_path + '/' + file.split('\\')[-2] + '/' + i):
                shutil.move(file, train_path + '/' + file.split('\\')[-2] + '/' + i)
            else:
                pass
    for dir in os.listdir(src_path):
        src = os.path.join(src_path, dir)
        current_len = len(os.listdir(src))
        test_count = math.ceil(current_len * ratio)
        # 测试集
        # print(type(os.listdir(src)))
        for i in os.listdir(src)[0:test_count]:
            file = os.path.join(src, i)
            if not os.path.exists(test_path + '/' + file.split('\\')[-2] + '/' + i):
                shutil.move(file, test_path + '/' + file.split('\\')[-2] + '/' + i)
            else:
                pass
        # 训练集
        for i in os.listdir(src)[test_count:current_len]:
            file = os.path.join(src, i)
            if not os.path.exists(train_path + '/' + file.split('\\')[-2] + '/' + i):
                shutil.move(file, train_path + '/' + file.split('\\')[-2] + '/' + i)
            else:
                pass


if __name__ == '__main__':
    src_path = '../row'
    dist_path = '../dataset_test'
    train_path = os.path.join(dist_path, 'training_data')
    test_path = os.path.join(dist_path, 'testing_data')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    split_data(src_path=src_path,
               train_path=train_path,
               test_path=test_path,
               ratio=0.1)
