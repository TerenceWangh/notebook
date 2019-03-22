图像分类
=======

## 图像分类主要面临的问题
1. **角度变化**：同样的对象在不同的角度拍摄出的照片不同。
2. **尺度变化**：同样的分类其实体大小可能有很大的差异。
3. **形状变化**：很多物体不是固定的形状，可能会有很多不同的形状。
4. **部分遮挡**：很多时候，我们研究的对象可能会被遮挡，只露出很小的一部分。
5. **光照条件**：光照的变化对识别也会产生重要的影响。
6. **背景差异**：被研究对象处在环境的背景可能跟自身很相似。
7. **类内差异**：同一个类可能会有很多不同的小分类，之间的差异比较大。

## 图像分类的流程
1.  **输入**：输入有多张(N)图片组成，每张图片被标记为分类(共K种分类)的一种。我们称这种数据为训练集。
2.  **学习**：我们的目标就是使用训练集去学习每个分类是什么样子的，称之为训练分类或者是学习一个模型。
3.  **验证**：最后，我们通过让分类器预测新的图片的分类来达到测试分类器的质量的目的。我们通过对比预测的分类和真实的分类来得到评估的指标。

## 最近邻分类
该分类和卷积神经网络没有什么关系，并且在实际应用中，该分类算法也不会用于图像分类中。但是，该分类算法能够给我们提供一些基本分类算法的思想。

### CIFAR-10数据集
[CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)是一个非常流行的用于分类的数据集，包括60000张32像素大小的小图片。每张图像包含10个分类中的一种类别。这60000张图片被分为50000张训练图片以及10000张测试图片。下图是截取的一部分：
{% asset_img nn.jpg %}
假设现在有50000张CIFAR-10的训练图片，每个分类有5000张图片。目标是去预测剩下的10000张图片。最近邻分类拿到一张测试图片时，会跟训练图像的每一张图片进行对比，并且给出与训练图片最接近的那个分类。在上图的右侧给出的测试数据的分类结果，在第8行数据中，与测试图像马最相近的图片是一个红色的车，可能是由相似的背景造成的，所以导致了错误的将马分类成了车。

### L1距离
*现在给出我们是如何对比两张图片的*：
给定两张32 x 32 x 3的图片 $I_1, I_2$，一个合理的选择是L1距离公式：$d_1(I_1, I_2) = \sum_p{|I_1^p - I_2^p|}$。
{% asset_img nneg.jpg %}
如上图是L1距离的具体计算过程，是比较每个像素的差值的绝对值，然后进行求和。如果两个图片十分相似，那么这个值将会趋向于0；如果两个图片差别非常大，那么这个值将会非常大。

***

**代码的实现**
1. 为了能够方便的进行训练, 加载我们所需的数据CIFAR-10到内存中。数据分为4个数组：*训练数据*/*训练标签*/*测试数据*/*测试标签*。
    ``` python
    Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/')
    # flatten out all images to one-dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  
    ```
    > 在上述代码中，Xtr是大小为50000 x 32 x 32 x 3的数组，即保存了50000张32x32像素大小的3个通道RGB图片。Ytr是一个50000大小的一维数组，保存着训练的标签，由于是10个分类，所以值为`0 - 9`
2. 下面给出训练数据以及验证的代码：
    ``` python
    # createa Nearest Neighbor classifier class.
    nn = NearestNeighbor()

    # train the classifier on the training images and labels.
    nn.train(Xtr_rows, Ytr)

    # predict labels on the test images.
    Yte_predict = nn.predict(Xte_rows) 
    # and now print the classification accuracy, which is the average
    # number of examples that are correctly predicted (i.e. label matches).
    print('accuracy: %f' % (np.mean(Yte_predict == Yte)))
    ```
    > 准确率是评估的标准之一，用来衡量预测是否正确。
3. 分类的代码：
    ``` python
    import numpy as np
    
    class NearestNeighbor(object):
        def __init__(self):
            pass
        
        def train(self, X, y):
            ''' X is N x D where each row is an example. Y is 1-dimension of size N. '''
            # nearest neighbor classifier simply remebers all the training data.
            self.Xtr = X
            self.Ytr = y
        
        def predict(self, X):
            ''' X is N x D where each row is an example we wish to predict label for.'''
            num_test = X.shape[0]
            # let's make sure that the output type matches the input type.
            Ypred = np.zeros(num_test, dtype=self.Ytr.dtype)

            # loop over all test rows.
            for i in xrange(num_test):
                # find the nearest training image to the i'th test image.
                # using the L1 distance (sum of absolute value differences).
                distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
                min_index = np.argmin(distances)    # get the index with smallest distance.
                Ypred[i] = self.Ytr[min_index]      # predict the label of the nearest example.
            
            return Ypred
    ```
    > `train(X, y)`函数输入为训练的数据和标签，`predict(X)`函数输入新的数据，并给出预测的标签。

4. 结论：上面的代码能够实现**38.6%**的准确率。

### L2距离
计算两个两个向量的距离有很多种方式。另一种常用的选择是**L2距离**，L2的距离在几何中的解释是欧几里得距离。距离的公式如下：
$$d_2(I_1, I_2) = \sqrt{\sum_p{(I_1^p - I_2^p)^2}}$$
在代码的实现中，只需要更改距离的计算公式：
``` python
    distances = np.sqrt(np.sum(np.square(self.Xtr - X[i. :]), axis=1))
```
但是L2距离的结果只有**35.4%**。

### L1和L2距离对比
一般而言，L2距离较L1距离更为严苛，因此L2距离更适用于中等分歧的场合。

## K-近邻分类
&emsp;&emsp;你可能发现仅仅使用最相近的图片得出标签是不靠谱的。因此，**k-近邻分类**是一种更靠谱的方式去预测标签。k-近邻分类的思想是，不仅仅是找出仅仅一张最相近的图像，而是找到最相近的k张图片，从中选出出现最多的那个标签。实际上，当k=1时，就是上面所述的最近邻分类。但是，较高的k能够平滑分类器的影响并增强分类器的鲁棒性。

**在实际使用中，你可能总是想使用k近邻分类，但是k的值如何确定呢？**
### 超参数调节
&emsp;&emsp;k近邻分类需要设置k的值，但是哪个数能够最好的工作？另外，我们还有很多距离函数的选择，比如**L1距离**，**L2距离**，还有其他的一些没有涉及的距离函数。这些选择成为**超参数**，在机器学习算法的设计中，这个概念经常出现，但是悬着哪个值或那种方法就比较难以确定了。
&emsp;&emsp;一个比较可行的方法是尝试不同的值来比较哪个方法的结果更好，但是**我们不能通过测试数据集来调节超参数**。最好的来获取该超参数的方法是使用以前从未出现过的数据集。否则，当你把数据移植到其他数据上时，你会发现性能会有很大的降低，这就是我们俗称的**过拟合**。
>因此，最好的方法是使用从没有进行训练过的测试集来调节超参数。常使用的方式是将训练数据划分为两个部分，分别是训练集和**验证集**。以之前的CIFAR数据为例，我们可以将训练数据划分为49000张训练图片，1000张验证图片。验证数据是为了能够有效的调节超参数。具体的代码如下：
``` python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50000 x 3702 matrix
Xval_rows = Xtr_rows[:1000, :]  # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :]   # take last 49000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    # use a particular value of k and evalution on validation data
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)

    # here we assume a modified NearestNeighbor class that can take a k as input
    Yval_predict = nn.predict(Xval_rows, k=k)
    acc = np.mean(Yval_predict == Yval)
    print('accuracy: %f' % acc)

    # keep track of what works on the validation set
    validation_accuracies.append((k, acc))
```

> **交叉验证** 当你的训练数据或者验证数据很小的时候，人们通常使用较为复杂的交叉验证的方式进行超参数调节。以之前为例，我们不再是选择1000个数据项作为验证集，其他的作为训练集。而是通过迭代不同的验证集并通过计算平均的性能来确定k的值。例如，5-fold交叉验证，我们将数据平分到5个桶内，用4个桶作为训练数据，剩下的1个桶作为验证集。我们通过迭代每个桶作为验证集来验证性能，最后将这些不同的验证结果进行平均。

{% asset_img cvplot.png %}
上图给出了5-fold交叉验证确定的k参数的结果。对于每个k值，我们使用4个桶进行训练，剩下的一个进行验证。因此，每个k值来说将会给出5个计算的精确度（y轴作为精确度）。图中的折线代表了精确度的走向。在这个例子中，交叉验证表明当k=7的时候，能够有较好的性能。如果使用更多的桶，那么得到的结果将会更加平滑，也就意味着有更小的噪声。

### 实践结果
&emsp;&emsp;由于交叉验证比较计算消耗比较昂贵，因此人们更加倾向于单一验证的划分并非交叉验证。人们一般使用**50%-90%**作为训练数据，其他的作为验证数据。这取决于很多因素：例如，如果超参数的数目非常多，你可能倾向于使用较大的验证集。如果给的数据量较少，使用交叉验证比较靠谱。典型的分桶的数量为`3, 5, 10`。

## 最近邻分类的优缺点
1. 优点：
    * 理论和实践都比较简单。
    * 训练不需要花费实践，所有的数据都被存储起来。
    * 
2. 缺点：
    * 在测试的时候需要花费大量的时间。
3. 总结：
    * **近似近邻算法(ANN)**能够改善测试的时间花销。这些算法通常是在精确度和空间/时间的复杂度上做权衡，通常的方法是在预处理或者索引阶段建立kdtree或者是使用k-means算法。
    * K近邻算法是一个比较好的算法，尤其是在低维度的数据中。但是在图像识别中用处不大。主要是因为图像是高维度的，并且高维空间的距离很多是跟视觉上差异很大的，比如当对图像进行平移、遮挡或者是调整亮度，在L2距离上将会产生很大的影响，使得相同的图片产生特别大的差异。
