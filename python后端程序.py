import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # 导入 Flask-CORS
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from predict1 import MyModel
app = Flask(__name__)
CORS(app)  # 启用 CORS

# ... 这里是你的 Flask 应用代码 ...
# 加载已经训练好的模型
model = MyModel()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 图像分类类别名称
with open('imagenet_classes.txt') as f:  # 替换成你的类别名称文件路径
    classes = [line.strip() for line in f.readlines()]


# 定义预测函数
def predict_image(image_path):
    img = Image.open(image_path)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度
    with torch.no_grad():
        output = model.forward(img_tensor)

    """一个类别"""
    # _, predicted = output.max(1)
    # predicted_class = classes[predicted.item()]
    """六个类别"""
    # 读取类别信息
    # with open('imagenet_classes.txt', 'r', encoding='utf-8') as file:
    #     classes = file.readlines()
    # classes = [c.strip() for c in classes]
    #
    # # 假设模型输出的数组为 prediction_array
    # prediction_array = output.squeeze().numpy()
    #
    # # 获取最相似的六个类别
    # top_similar_classes = sorted(range(len(prediction_array)), key=lambda i: prediction_array[i], reverse=True)[:6]
    # top_similar_classes_names = [classes[i] for i in top_similar_classes]
    # return top_similar_classes_names

    # """六个类别及其概率"""
    # 获取最接近的六个类别及其概率
    # 假设模型输出的原始概率值
    prediction_array = (1/(1+np.exp(-output.squeeze().numpy()))).astype(float).round(decimals=4)

    # 读取类别信息（不变）
    with open('imagenet_classes.txt', 'r', encoding='utf-8') as file:
        classes = file.readlines()
    classes = [c.strip() for c in classes]

    # 获取最接近的六个类别及其概率
    print(type(prediction_array), type(classes))
    top_similar_classes_indices = np.argsort(prediction_array)[-6:][::-1]
    top_similar_classes = [{'class': classes[i], 'probability': prediction_array[i]} for i in top_similar_classes_indices]
    print(top_similar_classes)
    return top_similar_classes







# 定义 Flask 路由，处理上传图片并进行预测
@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'})

    file = request.files['file']
    file.save('uploaded_image.jpg')  # 将上传的图片保存到服务器
    prediction_result = predict_image('uploaded_image.jpg')  # 调用预测函数
    print(prediction_result)
    # return jsonify({'prediction': prediction_result})
    return jsonify({'result': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)  # 运行 Flask 应用