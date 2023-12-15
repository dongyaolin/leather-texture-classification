"""
该脚本对采集的数据集进行数据增强
将一个文件夹下的图片进行数据增广后保存到目标文件夹。
如果要对所有数据集下各类别进行数据增广，需要分别对每个文件夹使用本脚本，具体实现使用当前文件夹下的dataaug_every.py脚本
"""

from PIL import Image, ImageOps, ImageEnhance
import os
import random


class ImageAugmenter:
    def __init__(self, folder_path, save_path, num):
        self.folder_path = folder_path
        self.save_path = save_path
        self.num = num
    def augment_images(self):
        # 遍历文件夹中的所有图片
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # 读取图片
                image_path = os.path.join(self.folder_path, filename)
                original_image = Image.open(image_path)

                # 对图片进行数据增强
                for i in range(self.num):
                    # 随机旋转
                    rotated_image = original_image.rotate(angle=random.randint(-30, 30))

                    # 随机裁剪
                    cropped_image = original_image.crop((random.randint(0, 50), random.randint(0, 50),
                                                         random.randint(200, 250), random.randint(200, 250)))

                    # 随机调整亮度和对比度
                    brightness_enhancer = ImageEnhance.Brightness(cropped_image)
                    contrast_enhancer = ImageEnhance.Contrast(cropped_image)
                    cropped_image = brightness_enhancer.enhance(random.uniform(0.5, 1.5))
                    cropped_image = contrast_enhancer.enhance(random.uniform(0.5, 1.5))

                    # 随机添加噪声
                    noise_image = cropped_image.convert('L').point(lambda x: random.randint(0, 100))

                    # 随机反转
                    inverted_image = ImageOps.invert(noise_image)

                    # 保存增强后的图片
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    rotated_image.save(os.path.join(self.save_path, 'rotated_image_{}_{}.jpg'.format(filename, i)))
                    cropped_image.save(os.path.join(self.save_path, 'cropped_image_{}_{}.jpg'.format(filename, i)))
                    inverted_image.save(os.path.join(self.save_path, 'inverted_image_{}_{}.jpg'.format(filename, i)))

                # 将所有增强后的图片统一大小并保存
                for i, filename in enumerate(os.listdir(self.save_path)):
                    if filename.startswith('rotated_image_{}'.format(filename)) or filename.startswith(
                            'cropped_image_{}'.format(filename)) or filename.startswith('inverted_image_{}'.format(filename)):
                        image = Image.open(os.path.join(self.save_path, filename))
                        resized_image = image.resize((224, 224))
                        resized_image.save(os.path.join(self.save_path, 'esized_image_{}_{}.jpg'.format(filename, i)))


if __name__ == '__main__':
    augmenter = ImageAugmenter('../row/class_1', '../row/images', 100)
    augmenter.augment_images()
