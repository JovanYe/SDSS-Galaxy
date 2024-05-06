import torch
import random
from ResNet18 import ResNet
import matplotlib.pyplot as plt
from dataProgressing import GalaxyDataset
from ResNet18 import ContrastiveLoss
from torch.utils.data import DataLoader
import torch.utils.data as Data

def create_contrastive_pairs(imageList, nums=512): # 每个类别，正对，负对各nums对
    pairs = []
    labels = []
    classNum = len(imageList)
    for i, subImgList in enumerate(imageList):
        indices = range(len(subImgList))
        numbers = list(range(classNum))
        numbers.remove(i)

        for k in range(nums):
        # 正对：同一类别的两个样本
            idx1, idx2 = random.sample(indices, 2)
            pairs.append((subImgList[idx1], subImgList[idx2]))
            labels.append(1)  # 正对

        # 负对：不同类别的两个样本
            idx1 = random.choice(indices)
            k = random.choice(numbers) # 从其余9个类别选择一个
            idx2 = random.choice(range(len(imageList[k]))) # 选择一个样本
            pairs.append((subImgList[idx1], imageList[k][idx2]))
            labels.append(0)  # 负对

    return pairs, labels


if __name__ == "__main__":

    dataset = GalaxyDataset()
    imageList = dataset.imageList(is_aug=True, min_size=512)
    # 初始化模型和优化器
    model = ResNet()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    # 训练循环
    num_epochs = 8
    Loss = []
    for epoch in range(num_epochs):
        contrastive_pairs, pair_labels = create_contrastive_pairs(imageList)
        print(len(contrastive_pairs))
        pair_dataset = [(x[0], x[1], y) for x, y in zip(contrastive_pairs, pair_labels)]
        # pair_dataset = Data.TensorDataset(contrastive_pairs, pair_labels)
        pair_dataloader = DataLoader(pair_dataset, batch_size=32, shuffle=True)

        running_loss = 0.0
        for data in pair_dataloader:
            img1, img2, label = data
            img1, img2, label = img1.to(device=device).float(), img2.to(device=device).float(), label.to(device=device).float()

            # 清除梯度
            optimizer.zero_grad()

            # 提取特征
            output1 = model(img1)
            output2 = model(img2)

            # 计算损失
            loss = criterion(output1, output2, label)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


        print(f"Epoch {epoch +1}, Loss: {running_loss / len(pair_dataloader)}")
        Loss.append(running_loss / len(pair_dataloader))

    torch.save(model, "./model/resNet_model.pkl")
    plt.plot(range(1,num_epochs+1), Loss, color='green', linestyle='solid', label='Train_Loss')
    plt.legend(loc='upper right')
    plt.title('Loss during ResNetModel training')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./' + str(num_epochs) + '.png')
    plt.close()

