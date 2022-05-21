# 机器学习course

## 第一章：引言

- 监督学习：
  - traditional supervised learning: 传统监督学习-每一个样本都有标签
    - 支持向量机support vector machine
    - 人工神经网络neural networks
    - 深度神经网络deep neural networks
  - unsupervised learning：非监督学习-所有训练数据都没有标签
    - 聚类（clustering
    - EM算法（Expectation-maximization algorithm
    - 主成分分析（Principle component analysis
  - semi-supervised learning：半监督学习-训练数据中一部分有标签一部分没有标签

- 按标签是否连续意义分为回归和分类

- 机器学习算法的过程：
  - 提取特征
  - 特征选择
  - 不同算法对特征空间进行曲线划分

## 第二章：支持向量机

- support vector machine算法：
  - 解决线性可分问题
    - 如果问题是线性可分的，则存在无数超平面将各个类别分开；但其中哪个是最好的呢
    - 基于最优化理论：对任意一条分割线，分别向两侧移动，当其擦到一些向量（支持向量）时停止，两条平行线间的距离叫做距离margin，我们要求的最优分割线是使间隔最大的一条线，且在两条平行线中间。 
    - 线性可分是指一个训练样本集{（xi，yi）}，存在w，b使当yi=+1时，wTxi+b>0.……
    - 点(x0,y0)到线w1x+w2y+b=0距离  $ d=\frac{\left |w1x0+w2y0+b\right | }{\sqrt{w1^{2} +w2^{2} } } $ 
    - 优化：用a缩放w b：(w,b) -> (aw,ab)；最终使支持向量x0上有|wTx0+b|=1，在非支持向量上有>1；所以：支持向量到超平面的距离将会变为$d=\frac{w^{T}x_{0}+b}{\left \| w \right \| } =\frac{1}{\left\| w\right\|}$；再所以：优化问题定为最小化$\frac{1}{2}\left\|w\right\|^2$, 限制条件：$y_i (w^Tx_i+b)>=1$,
  - 再将线性可分问题获得的结论推广到线性不可分的情况
    - 线性不可分情况是无解的，因此要放松限制条件：对每个训练样本和标签(xi,yi)，松弛变量$\delta_i$，限制条件改写为$y_i (w^Tx_i+b)>=1-\delta_i$，最小化$\frac{1}{2}\left\|w\right\|^2+C\sum_{i=1}^N\delta^2_i$：







# 李宏毅机器学习-学习笔记

## 第0节-相关规定

---

- 监督学习是有数据，有数据得标注，计算机来学习分类

- 无监督学习是没有标注，计算机自己学习分类

- 强化学习用于不知道要怎么标注，比如如何下围棋，人也不知道哪里最好，但是知道哪里好哪里不好

- 异常检测：当出现数据集之外的东西如何处理

- 可解释AI：告诉我们为什么这样分类

- 模型攻击：给AI出难题，用一些细小的变化去欺骗

- domain adaption：当数据与训练集不一样时

- 模型压缩：使模型简易化 

## 第1节-基本概念

---

- linear model定义步骤
  - 定义模型Model：y = wx<sub>1</sub>+b
  - 定义损失Loss: L(b,w)，其中MAE：mean absolute error；MSE：mean square error
  - 优化算法optimization：w<sup>* </sup>, b<sup>*</sup> = *arg* min L
    - 梯度下降的过程：
    ![image-20220415183749195](C:\Users\yangqi\AppData\Roaming\Typora\typora-user-images\image-20220415183749195.png)
  - 预测，改进（如线性模型中增加特征维度），最后test
  - 模型更加复杂：
    - 线性回归就一条直线，实际情况可能没这么简单，会有很多Hard Sigmoid，可以用一堆Sigmoid Function去逼近这一折线：
      ![image-20220415223511738](C:\Users\yangqi\AppData\Roaming\Typora\typora-user-images\image-20220415223511738.png)
    - 可以得到更加复杂的模型：用sigmoid逼近：这就是神经网络的一层：
      ![image-20220415223904488](C:\Users\yangqi\AppData\Roaming\Typora\typora-user-images\image-20220415223904488.png)
    - 

---

### Pytorch使用

#### What is Pytorch

- 一个python中的机器学习的框架

- 两点优势：

  - 在GPU上进行n维张量的计算

  - 用于训练深度神经网络的自动微分

#### Step1：LoadData
- torch.utils.data.Dataset ：存储样本数据以及预期值
- torch.utils.data.DataLoader：对数据进行打乱以及batch化
```
dataset = MyDataset(file)
dataLoader = DataLoader(dataset, batch_size, shuffle=True`#：train时时要True的，test时是要false
```
- Creating Tensor：
```
x = torch.tensor([1,-1],[-1,1])
x = torch.from_numpy(np.array([1,-1],[-1,1]))
x = torch.zeros([2,2])#shape
x = torch.ones([2,2])
```
- Common Operations：
```
x = a+b-c
x = y.pow(2)
x = y.sum()
x = y.mean()
x = x.transpose(0,1)#维度互换
x = x.squeeze(0)#消除第0维
x = x.unsqueeze(1)
w = torch.cat([x,y,z], dim=1)#把维度为2*1*3，2*2*3，2*3*3的张量合成一个2*6*3的
```
- Gradient Calculation
```
x = torch.tensor([[1.,0.],[-1.,1.]], requires_grad=True)
z = x.pow(2).sum()
z.backward()
x.grad
```
### Step2: Define Neutal Networks
- torch.nn - Network Layers
```
#torch.nn.Linear(in_features, out_features)
#比如32*64的全连接层b+wx是用64*32的w×32*1的x再加b得到64*1的y
layer = torch.nn.Linear(32, 64)
layer.weight.shape == torch.Size([64,32])
layer.bias.shape == torch.Size([64])
```
- 非线性激活函数
	- nn.Sigmoid()
	- nn.ReLU()
- 如何构建自己的神经网络
```
import torch.nn as nn
class Mymodel(nn.Module):#初始化你的模型，定义层
	def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10,32),
            nn.Sigmoid()
            nn.Linear(32,1)
        )
	def forward(self, x):#计算你的神经网络的结果
		return self.net(x)
```
#### Step 3:Loss函数
- Mean Squared Error均方误差用于回归任务：`criterion = nn.MSELoss()`
- Cross Entropy交叉熵用于分类任务：`criterion = nn.CrossEntropyLoss()`
- `loss = criterion(model_output, expected_value)`
#### Step 4优化参数
- Stochastic Gradient Descent随机梯度下降`optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0)`
- 对每一batch的数据
	- 调用optimizer.zero_grad()来重置模型参数的梯度
	- 调用loss.backward()来反向传播预测损失的梯度
	- 调用optimizer.step()来迭代模型参数
#### Step 5整个步骤
- dataset = MyDataset(file)----读取数据
- tr_set = DataLoader(dataset, 16,shuffle=True)---将dataset放入Dataloader
- model = MyModel().to(device)------构建模型，数据进设备
- criterion = nn.MSELoss()-----设置损失函数
- optimizer = torch.optim.SGD(model.patameter(),0.1)----设置优化器
- 训练过程：![image-20220417215506713](C:\Users\yangqi\AppData\Roaming\Typora\typora-user-images\image-20220417215506713.png)

- 验证过程：![image-20220417215659513](C:\Users\yangqi\AppData\Roaming\Typora\typora-user-images\image-20220417215659513.png)

- 测试过程：![image-20220417215851227](C:\Users\yangqi\AppData\Roaming\Typora\typora-user-images\image-20220417215851227.png)

- 训练模型存储导入：![image-20220417220230683](C:\Users\yangqi\AppData\Roaming\Typora\typora-user-images\image-20220417220230683.png)

### HW: Covid prediction

- objectives: 

  - To solve a regression problem with deep neural networks(DNN)

  - understand basic DNN train tips: hyper-parameter tuning; feature selection regularization.

  - get familiar with pytorch

- Task description:
  - COVID -19 cases prediction: given survey results in the past 3days in a specific state in US, then predict the percentage of new tested positive cases in the 3rd day.
  - data: surveys: 2700 samples
    - states(40, one hot vectors)
    - COVID-like illness(4)
    - behavior indicators(8)
    - mental health indicators(5)
    - tested positive cases(1)
  
   - test data: 893 samples
     - missing the prediction of the 3rd day

	- evaluation metric

 - root mean squared error(RMSE)

### 选修部分知识点

- 第一步找一个model，一个model就是一个function set就是设置好了一个神经网络架构，但是其中的参数还没有确定 
- 第二步，找一个function出来评判好坏
- 第三步找一个最好的function

关于预测宝可梦：

- Loss function是一个function的function，衡量一个model的好坏也就是一组w，b的好坏。
