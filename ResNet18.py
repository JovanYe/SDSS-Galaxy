import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 使用ResNet18作为特征提取器
class ResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet, self).__init__()
        # 加载预训练的 ResNet18 模型
        resnet = models.resnet18(pretrained=False)

        # # 冻结所有参数
        # for param in resnet.parameters():
        #     param.requires_grad = False

        # 提取 ResNet18 的特征提取部分
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # 定义自定义的全连接层，调整输出维度为 num_classes
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        # 前向传播
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 定义对比学习的损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, label):
        euclidean_distance = F.pairwise_distance(input1, input2, keepdim=True)
        loss = torch.mean(label * euclidean_distance + (1 - label) * F.relu(self.margin - euclidean_distance))
        return loss

if __name__ == "__main__":
    pass