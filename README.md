# 皮革纹理分类
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.9.0-%237732a8)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-v2.0-%2348c35e)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.5.3-%23fefefe)](https://opencv.org/)

### 皮革纹理分类 
### 项目简介 
[![基于深度学习](https://img.shields.io/badge/基于-深度学习-blueviolet)](深度学习链接)
[![高效分类系统](https://img.shields.io/badge/高效分类系统-提供-orange)](分类系统链接)
[![多用途应用](https://img.shields.io/badge/多用途-适用-success)](多用途链接)

这个项目基于深度学习技术，旨在识别和分类各种皮革纹理。无论您是从事时尚设计、工业制造还是艺术创作，这个项目将为您提供一个高效的纹理分类系统。



<details>

<summary>功能特性 </summary>


[![图像识别](https://img.shields.io/badge/图像识别-支持-brightgreen)](功能详情链接)
[![多类别分类](https://img.shields.io/badge/多类别分类-支持-blue)](功能详情链接)
[![模型训练和优化](https://img.shields.io/badge/模型训练和优化-提供训练脚本-red)](功能详情链接)
[![易用性](https://img.shields.io/badge/易用性-简洁易用-yellow)](功能详情链接)

- **图像识别分类**：使用深度学习模型对皮革图像进行分类，识别各种皮革类型，如皮质、人造皮革、动物皮革等。  
- **多类别分类**：支持多个类别的皮革分类，提高系统对不同种类皮革的准确性和普适性。  
- **模型训练和优化**：提供训练脚本和工具，以便用户能够根据自己的数据集对模型进行训练和优化。   
- **易用性**：提供简洁易用的 API 和界面，使其他开发者能够轻松地集成此分类系统到自己的应用中。  

</details>

<details>
<summary>技术栈</summary>

[![PyTorch](https://img.shields.io/badge/PyTorch-v1.9.0-%237732a8)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-v2.0-%2348c35e)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.5.3-%23fefefe)](https://opencv.org/)
[![构建状态](https://img.shields.io/badge/构建状态-passing-brightgreen)](构建状态链接)

- **深度学习模型**：采用 Pytorch 搭建卷积神经网络 (CNN) 模型进行图像分类。  
- **Python 后端**：使用 Flask 框架构建简单易用的后端 API。  
- **图像处理工具**：使用 OpenCV 等库进行图像处理和预处理。

</details>

### 使用方法

1. - [x] **数据准备**：准备包含各种皮革图像的数据集，并按类别进行组织。[![数据准备](https://img.shields.io/badge/详细说明-orange)](https://www.pullywood.com/ImageAssistant/)
- 将收集到的原始数据按照皮革纹理类别放入文件夹row。如果同一类别图像在不同文件夹下，可通过scipts中的getimggromdir将同一类别图像放在一个文件夹内。
- 可以使用rm_duplicates.py对图像进行去重。
- 将收集到的原始数据重命名，可使用scripts中的rename.py进行重命名。
- 使用dataaug_every.py脚本对每一类别进行基础数据增强，该过程也可以在训练过程的dataset.py进行。
- 使用splitdataset.py脚本对原始分类好的数据集进行训练集和测试集划分。并放入identify_dataset文件夹下。注意：划分比例不同可能导致部分数据不能完全划分到训练集或测试集。
2. - [x] **模型训练**：使用提供的训练脚本对数据集进行训练，或者根据自己的需求修改并训练模型。![模型训练](https://img.shields.io/badge/使用指南-red)
   - 准备完分类好的数据集后。 使用:
   - ```python
     python train1.py
   - 训练好的模型位于models1文件夹下。
3. - [ ] **启动后端**：运行后端服务，提供图像分类的 API。![后端服务](https://img.shields.io/badge/后端服务-启动指南-blue)
4. - [ ] **API 调用**：通过 API 发送图像并接收分类结果。[![API 调用](https://img.shields.io/badge/API%20调用-示例代码-lightgrey)](API调用链接) 
### 贡献指南 [![提交 Issue](https://img.shields.io/badge/提交-Issue-9cf)](提交Issue链接) [![Pull Request](https://img.shields.io/badge/Pull%20Request-贡献代码-brightgreen)](PR链接)

欢迎任何形式的贡献和反馈！您可以通过提交 Issue 或 Pull Request 的方式参与项目，帮助改进代码质量、增加新功能或修复 Bug。

### 版权和许可 [![版权](https://img.shields.io/badge/版权-Duncan_Dong-orange)](版权链接) [![许可证](https://img.shields.io/badge/许可证-Apache%202.0-green)](许可证链接)

该项目采用 Apache 许可证 2.0。
