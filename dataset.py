from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataSet:
    def __init__(self, root_dir, batch_size, shuffle, num_workers, istrainning):
        super(DataSet, self).__init__()
        self.istrainning = istrainning
        self.dataset = datasets.ImageFolder(root=root_dir,
                                            transform=self.get_transforms())
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else num_workers * batch_size
        )

    def get_transforms(self):
        if not self.istrainning:
            return transforms.Compose([
                
                transforms.Lambda(self.convert_img),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224), antialias=True)
            ])
        else:
            return transforms.Compose([
                
                transforms.Lambda(self.convert_img),
                transforms.Grayscale(),
                # 随机水平翻转输入图像
                transforms.RandomHorizontalFlip(p=0.4),
                # 随机改变图像的亮度、对比度、饱和度和色调等属性，从而增加数据样本的多样性
                transforms.ColorJitter(),
                # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为指定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小）
                # 默认scale = (0.08, 1.0)
                transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
                transforms.ToTensor()
            ])

    @staticmethod
    def convert_img(img):
        return img.convert("RGB")

    def __len__(self):
        return len(self.dataset.imgs)

    def __iter__(self):
        for data in self.loader:
            yield data


if __name__ == '__main__':
    batch_size = 8
    num_workers = 0
    train_dataset = DataSet('./identify_dataset/training_data', batch_size, True, num_workers)
    test_dataset = DataSet('./identify_dataset/testing_data', batch_size, False, num_workers)
    # print(len(train_dataset))
    # print(len(test_dataset))
    # print(len(test_small_dataset))
    for inputs, labels in train_dataset:
        # print(inputs.shape)
        # print(labels.shape)
        print(labels[0].item())
