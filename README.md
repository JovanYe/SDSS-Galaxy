# SDSS-Galaxy
This is a ten category image classification task on Galaxy10 SDSS.

#ResNet18
这是一个基于ResNet18作为特征提取上游模块的对比学习方法，基于对比学习的损失函数进行训练，即由正对损失和负对损失共同构成。

#dataProcessing
这是一个图像数据处理模块，由于Galaxy 10的类别数量极不平衡，因此我们采用合适的样本增广方法。包括对图像的：随机角度旋转，随机水平和垂直翻转，加入随机噪声，以及随机缩放等操作对样本量少的类别进行增广处理。

#train
这是训练模块，包括为每一个epoch制作正负对数据集，我们采用的是随机采样正负对的方式，集大程度上保证了样本的均衡。但训练难度比较大，目前结果待更新...
