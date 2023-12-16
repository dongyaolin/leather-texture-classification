from dataaug import ImageAugmenter
import os

src = '../identify_dataset/training_data'
for root, dirs, files in os.walk(src):
    augmenter = ImageAugmenter(root, root, 10)
    augmenter.augment_images()