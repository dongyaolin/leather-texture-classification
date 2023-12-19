from flask import Flask, request, jsonify
from flask_cors import CORS  # 导入 Flask-CORS
import torch
from torchvision import models, transforms
from PIL import Image
app = Flask(__name__)
CORS(app)  # 启用 CORS

# ... 这里是你的 Flask 应用代码 ...
# 加载已经训练好的模型
model = models.resnet18(pretrained=True)
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        output = model(img_tensor)
    _, predicted = output.max(1)
    predicted_class = classes[predicted.item()]
    return predicted_class


# 定义 Flask 路由，处理上传图片并进行预测
@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'})

    file = request.files['file']
    file.save('uploaded_image.jpg')  # 将上传的图片保存到服务器
    prediction = predict_image('uploaded_image.jpg')  # 调用预测函数
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)  # 运行 Flask 应用