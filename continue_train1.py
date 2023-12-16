import torch
import torchvision.models as models
import torch.optim as optim

# 加载预训练模型，假设是一个 ResNet50
model = models.resnet50(pretrained=True)

# 冻结模型的一部分，比如不修改预训练的卷积层参数
for param in model.parameters():
    param.requires_grad = False

# 替换模型的最后一个全连接层（假设最后一层是全连接层）
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # 假设有10个新的输出类别

# 定义新的优化器，通常使用较小的学习率
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 加载之前的训练权重
checkpoint = torch.load('path_to_your_pretrained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# 继续训练模型
# 请注意：这里的代码假设你有一个新的数据集来继续训练模型
# 假设 train_loader 是你的新数据集的 DataLoader
for epoch in range(epoch, num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = your_loss_function(output, target)
        loss.backward()
        optimizer.step()

# 保存继续训练后的模型
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'path_to_save_continued_training_model.pth')
