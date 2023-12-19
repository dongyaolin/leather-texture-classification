from flask import Flask, request, jsonify, render_template
from predict1 import MyModel  # 导入你的模型类
import os

app = Flask(__name__)

# 加载预训练模型
# 假设模型是已经训练好的 PyTorch 模型
model = MyModel()


@app.route('/')
def index():
    return render_template('index.html')


# 接收上传的视频并进行预测
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # 确保上传的是视频文件
        allowed_extensions = {'mp4', 'avi', 'mov'}  # 可接受的视频文件扩展名
        filename = file.filename
        if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file format'})

        # 保存上传的视频文件
        video_path = os.path.join('uploads', filename)
        file.save(video_path)

        # 进行模型处理（假设 model.predict 可以处理视频路径）
        output = model.predict(video_path)

        # 处理模型输出并返回预测结果
        # 这里假设模型输出是一个字符串
        return jsonify({'prediction': output})


if __name__ == '__main__':
    app.run(debug=True, port=3345)
