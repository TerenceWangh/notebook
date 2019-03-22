神经网络实例
===========

## 生成数据
首先产生一个线性不可分的稀疏数据，代码如下：
``` python
N = 100     # number of points per class
D = 2       # dimensionality
K = 3       # number of classes
X = np.zeros((N * K, D))  # data matrix, each row = single example
y = np.zeros(N * K, dtype='uint8')
for j in range(K):
  ix    = range(N * j, N * (j + 1))
  r     = np.linspace(0.0, 1, N)
  t     = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N)*0.2
  X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
  y[ix] = j
# visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()
```
![线性不可分数据](spiral_raw.png)
> 通常都会数数据集进行预处理，将每个维度的特征的均值和方差处理成0。但是上面给出的数据，数据是从-1到1，并且分布较为均匀，因此我们跳过这一步。

## 训练Softmax线性分类
### 初始化参数
Softmax分类器使用线性的评价函数，并使用交叉损失熵作为损失函数。线性分类的参数包括权重矩阵`W`和偏置向量`b`。下面将这些参数进行随机初始化：
``` python
# initialize parameters randomly
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))
```
一定要注意我们有`D = 2`个维度并且是`K = 3`个分类。
### 实现评价函数
由于我们使用一个线性的分类器，因此可以简单的使用矩阵乘法并行的计算所有分类的评分：
``` python
# compute class scores for a linear classifier
scores = np.dot(X, W) + b
```
上面给出的数据集中共有300个点，因此上面计算得出的数组`scores`将会变成[300 x 3]，每一行代表3个分类的评分。
### 计算损失
第二个关键的部分是损失函数，它是一个可微分的目标，用来量化计算出评分和实际之间的差距。也就是说我们希望正确的分类的评分能够明显的高于其他分类的评分。有很多的损失函数可供选择，但这个例子，我们选用交叉损失作为损失函数。对于一个样本来说，$f$是计算的来的分类得分的矩阵，Softmax分类器计算的损失如下：
$$ L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) $$
可以看出Softmax分类器是对$f$中对三个分类非标准化的对数几率。然后对他们进行归一化得到概率。因此，表达式中$\log$是正确分类的概率。现在解释一下该损失函数是如何工作的：上式的表达式中，结果介于0和1之间。当正确分类的概率很小的时候，损失将会趋向于正无穷。反过来，当正确分类的概率很大的时候(接近1)，损失函数将会趋向于0，因为$\log(1) = 0$。因此，表达式$L_i$当正确分类很高的时候结果会很低，相反当概率很低的时候结果会很高。
然后将损失函数扩展到多个样本上并加入惩罚函数：
$$ L =  \underbrace{ \frac{1}{N} \sum_i L_i }_\text{data loss} + \underbrace{ \frac{1}{2} \lambda \sum_k\sum_l W_{k,l}^2 }_\text{regularization loss} $$
对于给定的评分数组`scores`，按照上述的方式进行计算，得到损失。首先获得概率的方法如下：
``` python
num_examples = X.shape[0]
# get unnormalized probabilities.
exp_scores = np.exp(scores)
# normalize them for each example.
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
```
得到的`probs`数组大小为[300 x 3]，每一行包含了三个分类的概率。特别的，我们对每行数组都进行了归一化，因此每行数据的和为1。现在我们能够求出正确分类的对数几率：
``` python
correct_logprobs = -np.log(probs[range(num_examples)], y)
```
`correct_logprobs`是只含有正确分类概率的一维数组。最后计算出平均的对数几率并加入惩罚损失：
``` python
# compute the loss: average cross-entropy loss and regularization
data_loss = np.sum(correct_logprobs) / num_examples
reg_loss  = 0.5 * reg * np.sum(W * W)
loss      = data_loss + reg_loss
```
上面的代码中，惩罚强度$\lambda$使用`reg`参数定义。最开始，使用随机初始化的参数会得到损失`loss = 1.1`，代表`np.log(1.0 / 3)`，这是因为初始化的权重很小，所有分类的概率大概是`1.0 / 3`。我们的目标是让损失函数尽可能的低，越接近于0越好，那么正确分类的概率就会越高。
### 使用反向传播计算梯度
我们已经定义了评价损失的方法，现在我们就要想法设法的取最小化它。我们将会使用*梯度下降*的方法。最开始使用随机初始化的参数，并使用损失函数进行参数梯度的验证，因此我们应该知道如何改变参数能够降低损失。引入中间变量`p`，是归一化概率的向量。对于一个样本来说，损失函数如下：
$$ p_k = \frac{e^{f_k}}{ \sum_j e^{f_j} } \hspace{1in} L_i =-\log\left(p_{y_i}\right) $$
我们现在希望能够了解$f$中计算得来的评分应该如何改变以降低损失$L_i$。换句话说就是希望能够推导出梯度$\partial L_i / \partial f_k$。损失$L_i$依赖于$p$，而$p$依赖于$f$。因此我们能够容易的使用链式法则来推导出梯度，实践也证明这是十分简单而且很容易理解的。那么最终的梯度为：
$$ \frac{\partial L_i }{ \partial f_k } = p_k - \mathbb{1}(y_i = k) $$
上面的表达式十分简洁明了。假设我们计算得来的概率为`p = [0.2, 0.3, 0.5]`，正确的分类是中间的那个。根据上面的推导，计算评分的梯度为`df = [0.2, -0.7, 0.5]`。回顾一下梯度的解释，那么这个结果十分的明了：增加第一个或最后一个元素的评分（错误的分类）将会导致损失函数的增加，而增加损失代表预测是更不准确的。当增加正确分类的评分则对损失是负影响的，也就是说-0.7的梯度告诉我们加大正确分类的评分将会导致损失函数的降低。
上面提到过`probs`保存了所有样本对于每个分类的概率，那么获得所有评分的梯度`dscores`的代码如下：
``` python
dscores = probs
dscores[range(num_examples), y] -= 1
dscores /= num_examples
```
最后，我们使用公式`scores = np.dot(X, W) + b`，因此我们只需要`scores`的梯度，也就是`dscores`，现在就可以用反向传播了：
``` python
dW = np.dot(X.T, dscores)
db = np.sum(dscores, axis=0, keepdims=True)
dW += reg * W # don't forget the regularization gradient
```
从上面的代码中，可以看到只需要进行简单的矩阵乘法操作，并加入惩罚项。注意到惩罚的梯度项的公式十分简单：`reg * W`，这是因为我们在损失函数上乘了一个常数`0.5` （$\frac{d}{dw} ( \frac{1}{2} \lambda w^2) = \lambda w$)，对于梯度计算这是十分便捷的。
### 进行梯度更新
计算出来了梯度，现在我们开始进行梯度的更新操作：
``` python
# perform a parameter update
W += -step_size * dW
b += -step_size * db
```
### 将上述所有工作组合起来
将上面所有工作进行组合，并使用梯度下降训练Softmax分类器：
``` python
# Train a Linear Classifier

# initialize parameters randomly
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(200):
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b 
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  correct_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(correct_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  if i % 10 == 0:
    print("iteration %d: loss %f" % (i, loss))
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples), y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg * W # regularization gradient
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
```
运行上面的代码，得到输出：
```
iteration 0: loss 1.096956
iteration 10: loss 0.917265
iteration 20: loss 0.851503
iteration 30: loss 0.822336
iteration 40: loss 0.807586
iteration 50: loss 0.799448
iteration 60: loss 0.794681
iteration 70: loss 0.791764
iteration 80: loss 0.789920
iteration 90: loss 0.788726
iteration 100: loss 0.787938
iteration 110: loss 0.787409
iteration 120: loss 0.787049
iteration 130: loss 0.786803
iteration 140: loss 0.786633
iteration 150: loss 0.786514
iteration 160: loss 0.786431
iteration 170: loss 0.786373
iteration 180: loss 0.786331
iteration 190: loss 0.786302
```
经过上面的训练，我们来测试一下准确率：
``` python
# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
```
上面输出为49%左右，效果不是很好，毕竟这是一个线性不可分的任务。下面给出学得的边界：
![线性分类边界](spiral_linear.png)

## 使用神经网络进行训练
很显然，线性分类器并不适合上面的数据集。下面将会使用神经网络进行训练。另外新加一个隐藏层即可满足上面的数据集。因此，我们需要两个权重和偏置的参数集(第一层和第二层)：
``` python
# initialize parameters randomly
h  = 100 # size of hidden layer
W1 = 0.01 * np.random.randn(D, h)
b1 = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))
```
计算评分的部分做一下调整：
``` python
# evaluate class scores with a 2-layer Neural Network
hidden_layer = np.maximum(0, np.dot(X, W1) + b1) # note, ReLU activation
scores       = np.dot(hidden_layer, W2) + b2
```
上面仅仅加了一行代码，下计算隐藏层，在此基础上计算最终的评分。最重要的是我们加入了一个非线性的部分，该代码中是加入了ReLU的激活函数。
剩下的部分也是像之前的那样，根据评分计算损失，然后计算评分的梯度`dscores`。但是，反向传播有所改变：首先反向传播到神经网络的第二层，看起来跟Softmax分类器很像：
``` python
# backpropate the gradient to the parameters
# first backprop into parameters W2 and b2
dW2 = np.dot(hidden_layer.T, dscores)
db2 = np.sum(dscores, axis=0, keepdims=True)
```
到这里还没有结束，因为`hidden_layer`是另一些参数和输入的函数。我们需要继续反向传播这些变量。计算如下：
``` python
dhidden = np.dot(dscores, W2.T)
```
现在，计算得出了隐藏层输出的梯度。接着，将会反向传播到非线性的ReLU激活函数上。因为$r = \max(0, x)$，因此有$\frac{dr}{dx} = 1(x > 0)$。结合链式法则，我们能够得出ReLU单元在输入大于0的时候直接将梯度传播过去，但是当输入小于0的时候，会切断传播。那么，对于ReLU的反向传播能够简单地使用下面代码实现：
``` python
# backprop the ReLU non-linearity
dhidden[hidden_layer <= 0] = 0
```
接着反向传播到第一层的权重和偏置：
``` python
# finally into W, b
dW1 = np.dot(X.T, dhidden)
db1 = np.sum(dhidden, axis=0, keepdims=True)
```
得到了梯度`dW1, db1, dW2, db2`，下面就能够进行参数的更新了。整个的代码看起来是十分相似的：
``` python
# initialize parameters randomly
h   = 100 # size of hidden layer
W1  = 0.01 * np.random.randn(D, h)
b1  = np.zeros((1, h))
W2  = 0.01 * np.random.randn(h, K)
b2  = np.zeros((1, K))

# some hyperparameters
step_size = 1e-0
reg       = 1e-3   # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):
  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(X, W1) + b1)  # note, ReLU activation
  scores = np.dot(hidden_layer, W2) + b2

  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

  # compute the Loss: average cross-entropy loss and regularization
  correct_logprobs = -np.log(probs[range(num_examples), y])
  data_loss        = np.sum(correct_logprobs) / num_examples
  reg_loss         = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
  loss             = data_loss + reg_loss
  if i % 1000 == 0:
    print("iteration %4d: loss %f" % (i, loss))

  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples), y] -= 1
  dscores /= num_examples

  # backpropate the gradient to the parameters
  # first backprop into parameter W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W1, b1
  dW1 = np.dot(X.T, dhidden)
  db1 = np.sum(dhidden, axis=0, keepdims=True)

  # add regularization gradient contribution
  dW2 += reg * W2
  dW1 += reg * W1

  # perform a parameter update
  W1 += -step_size * dW1
  b1 += -step_size * db1
  W2 += -step_size * dW2
  b2 += -step_size * db2
```
上面的代码输出为：
```
iteration    0: loss 1.098744
iteration 1000: loss 0.294946
iteration 2000: loss 0.259301
iteration 3000: loss 0.248310
iteration 4000: loss 0.246170
iteration 5000: loss 0.245649
iteration 6000: loss 0.245491
iteration 7000: loss 0.245400
iteration 8000: loss 0.245335
iteration 9000: loss 0.245292
```
测试精确率：
``` python
# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print("training accuracy: %.2f" % (np.mean(predicted_class == y)))
```
将会得到98%的精确率。对边界进行可视化为：
![神经网络训练的边界](spiral_net.png)

## 总结
我们已经引入了一个2维的数据集，并使用线性分类和2层神经网络进行训练。可以看出从线性分类到神经网络的代码修改时非常小的，包括评分函数的改变，反向传播形式的变化。
