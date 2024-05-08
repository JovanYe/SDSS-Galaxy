# SDSS-Galaxy
This is a ten category image classification task on Galaxy10 SDSS.

#ResNet18
这是一个基于ResNet18作为特征提取上游模块的对比学习方法，基于对比学习的损失函数进行训练，即由正对损失和负对损失共同构成。

#dataProcessing
这是一个图像数据处理模块，由于Galaxy 10的类别数量极不平衡，因此我们采用合适的样本增广方法。包括对图像的：随机角度旋转，随机水平和垂直翻转，加入随机噪声，以及随机缩放等操作对样本量少的类别进行增广处理。

#train
这是训练模块，在每一个epoch我们都会基于随即采样的方式制作正负对数据集，极大程度上保证了样本的均衡。

训练的结果显示，在选择较大batch_size时，模型几乎无法收敛，loss卡在0.5无法继续下降。
![64](https://github.com/JovanYe/SDSS-Galaxy/assets/162402413/fb4fb2a9-1c96-44e1-886d-df55772af9a9)


若一对一对进行训练（batch_size=1)，则模型的loss可收敛到0.27左右。目前暂不清楚是什么原因。
![16](https://github.com/JovanYe/SDSS-Galaxy/assets/162402413/65ad93e2-2829-4636-ac6b-719535404c37)

