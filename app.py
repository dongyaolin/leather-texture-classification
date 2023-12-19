from flask import Flask, request, jsonify, render_template
from PIL import Image
from predict1 import MyModel

app = Flask(__name__)

# 加载预训练模型
# 假设模型是已经训练好的 PyTorch 模型
model = MyModel()


@app.route('/')
def index():
    return render_template('index.html')


# 接收上传的图片并进行预测
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image = Image.open(file.stream)
        # image_tensor = preprocess_image(image)
        output = model.predict(image)

        # 处理模型输出并返回预测结果
        # 这里假设模型输出是一个字符串
        return jsonify({'prediction': output})


if __name__ == '__main__':
    app.run(debug=True, port=3345)
