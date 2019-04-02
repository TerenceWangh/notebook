You Only Look Once
===

YOLOv1 arXiv: <https://arxiv.org/abs/1506.02640>

YOLOv2 arXiv: <https://arxiv.org/abs/1612.08242>

YOLOv3 arXiv: <https://arxiv.org/abs/1804.02767>


背景
---
首先，对于之前的R-CNN系列的检测方法，是多个步骤的：
1. 在图片中产生预选框；
2. 对每个预选框进行分类；
3. 对预选框进行提炼，去除重复的检测，并对预选框进行重新评分。

由于上述的步骤比较繁杂，因此对多阶段的检测任务而言，需要对每一步进行调优，便产生了调优困难的问题。
因此，YOLO产生了，YOLO使用单一的卷积网络，同时预测边界框和类别的可能性，是一种端到端的网络。YOLO主要具有以下特点：
1. 速度快，能达到实时的要求。由于去除了繁杂的任务，YOLO只需要端到端的训练和测试步骤，使得YOLOv1能够达到45fps，快速的版本则能够达到150fps。
1. 预测的时候，关注的是整张图像的特征，而不像滑动框那样仅仅是关注图像的某部分特征，因此能够包含更加丰富和全面的纹理特性。
1. YOLO能够学习目标的更加一般性的表示，能够更加宽泛的适合更多种类的图像。
1. YOLO在准确率上是较two-stage的算法低的，主要体现在边界框的准确度，尤其是小物体。

YOLO v1
---

### 框架
YOLO将图像分成$S \times S$的网格，每个网格给出$B$个候选框并针对每个候选框给出置信值。置信值反映了候选框包含目标的可能性以及候选框边界预测的准确程度。置信度定义为：$Pr(Object) \times {IOU}_{pred}^{thuth}$，也就是说当网格内不存在目标的时候，置信度为0，存在目标时，置信度为`IOU`。其实每个网格只负责目标的中心点落在该网格的目标。
每个候选框由5个参数构成：$(x, y, w, h, confidence)$。$(x, y)$表示相对于网格的中心点位置，$(w, h)$表示相对于整张图像的宽和高的比例。因此，这4个参数的值都是`0到1`之间。$confidence$表示候选框和任何ground truth的`IOU`。
另外，每个网格还给出`C`个分类的概率：$ Pr({Class}_i|Object) $，代表该网格存在该分类目标的可能性。在测试的时候，我们将其与上面的置信度相乘：$ Pr({Class}_i|Object) \times Pr(Object) \times {IOU}_{pred}^{thuth} = Pr({Class}_i) \times {IOU}_{pred}^{thuth} $，得到了对每个候选框，分类存在的可能性。
> 也就是说，将图像先分成$S \times S$的网格。然后每个网格预测出$B$个候选框(每个候选框有5个参数，表示位置和置信度)。同时，每个网格都给出$C$个分类存在的可能性。因此最终的输出为$S \times S \times (B \times 5 + C)$。参考下图

![网格化](YOLO/model.png)

网络的结构如下：
![网络结构](YOLO/framework.png)

网络使用$S = 7, B = 2, C = 20$来测试`PASCAL VOC`的数据集。
由于每个网格相当于只负责目标的中心点落在该网格的目标，而参数$B = 2$，也就是每个网格产生两个候选框，因此多目标中心点落在同一个网格中，效果会很差。同时，小的目标检测也会有很多的问题。

### 训练过程
1. 训练网络的前20个卷积层：仅仅使用前20个卷积层，后面另外加上`average-pooling layer`和一个`fully connected layer`方便进行训练。
1. 使用训练好的前20个卷积层，如上述的网络结构后面加上4个`convolutional layer`和2个`fully connected layer`，进行总体的训练。

### 更多细节
1. 激活函数的使用：
  最后一层使用线性激活函数，其他层使用`Leaky-Relu`:
  $$
    \phi (x) =
    \begin{cases}
      x & x > 0 \\
      0.1x & x <= 0
    \end{cases}
  $$
1. 损失函数设计:
  $$
  \begin{equation}
  \begin{split}
    & \lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^B{\mathbb{1}_{ij}^{obj}[{(x_i-\hat{x}_i)}^2+{(y_i-\hat{y}_i)}^2]}\\
    & +\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^B{\mathbb{1}_{ij}^{obj}[{(\sqrt{w_i}-\sqrt{\hat{w}_i})}^2+{(\sqrt{h_i}-\sqrt{\hat{h}_i})}^2]}\\
    & +\sum_{i=0}^{S^2}\sum_{j=0}^B{\mathbb{1}_{ij}^{obj}{(C_i-\hat{C}_i)}^2}\\
    & +\lambda_{noobj}\sum_{i=0}^{S^2}\sum_{j=0}^B{\mathbb{1}_{ij}^{noobj}{(C_i-\hat{C}_i)}^2}\\
    & +\sum_{i=0}^{S^2}{1}_{i}^{obj}\sum_{c \in classes}{{(p_i(c)-\hat{p}_i(c))^2}}
  \end{split}
  \end{equation}
  $$
  $\mathbb{1}_{i}^{obj}$表示第i个网格中有目标的情况下有效，而$\mathbb{1}_{ij}^{obj}$表示第i个网格的第j个推荐框有目标的情况下有效。
1. 学习率的设置策略：
    * 在前几个epochs的时候缓慢的将学习率从`1e-3`增加到`1e-2`，来防止学习率过大导致梯度不稳定。
    * 使用`1e-2`的学习率训练75个epochs。
    * 使用`1e-3`的学习率训练30个epochs。
    * 使用`1e-4`的学习率训练30个epochs。
1. 使用了dropout层来防止过拟合。