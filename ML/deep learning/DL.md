[toc]



Deep feedforward networks

/MLP

$y = {f}^{\star}(x), \quad \bold{y} = f(x; \theta)$ 学习出$\theta$ 来获取最好的函数近似



大多数参数话模型定的是一个的分布，模拟现实分布

$p(y|x;\theta)$ 使用最大似然，即模型预测和训练数据的交叉熵是cost function

$J(\theta) = -\bold{E}_{x,y~\hat{p}_{data}}logp_{model}(y|x)$

交叉熵损失函数更为通用，而MAE、MSE的某些地方梯度太小不便于学习。

> 需要激活函数的原因
>
> 在分类问题中，将输出转化为一个概率[0, 1]， 将涉及最大化最小化归一
>
> 而这些操作会带来梯度消失的问题，梯度消失就没有了学习的方向

$sigmoid(1)= \frac{e^x}{1+e^x},\\ softmax(x_i)=\frac{e^{x_i}}{\Sigma_je^{x_j}}, log\;softmax(x_i) = x_i-log\Sigma_je^{x_j}$

softmax 具有常量稳定性，即大家同加一个常量没有影响, 所以总是最大值成为那个softmax 的输出



### 隐藏层

> - $ReLU$       $g(z)=max\{0, z\}$
>
>   默认较好的激活函数
>
>   z = 0处不可导
>
>   - 泛化: $g(z) = max(0,z)+ɑ_i min(0,z)$
>   - $leakey\;ReLU$: $max\{0, z\}+0.01min\{0, z\}$
>   - $maxout$ 
>
> - $Sigmoid, tanh$
>
>   - $Sigmoid: g(z)=\sigma(z)$
>   - $tanh(z)=2\sigma(2z)-1$
>
>   这类激活函数 接近边界时，容易进入saturation，导致梯度为0，难以学习
>
>   tanh一般会比sigmoid表现更好
>
> - 
>
>   - 在MINIST的图像识别上，使用 $h=cos(W+b)$ 使得正确率更高
>
>   - 直接用线性单元作为隐藏层
>
>   - softmax
>
>   - $Radial\;basis\;function(RBF): h_i=exp(-\frac{1}{\sigma^2_i} ||\mathbf{W}_{:,i}-x||^2)$
>
>     这个函数对于大多数x直接饱和至0，较难优化
>
>     <img src="/Users/cxw/Learn/3_Coding/Typora/IMG/image-20210922174206069.png" alt="image-20210922174206069" style="zoom:33%;" />
>
>   - $Softplus: g(a)=\zeta(a)=log(1+e^a)$
>
>     <img src="/Users/cxw/Learn/3_Coding/Typora/IMG/image-20210922174437637.png" alt="image-20210922174437637" style="zoom: 33%;" />
>
>     极不推荐，效果比ReLU差

### 网络结构

**确定网络的深度和每个layer的宽度**

> 一定可以通过一个大的MLP找到想要的函数
>
> 找不到的原因 1. 找不到参数 2. 由于过拟合找错了函数

**更深的模型能够减少泛化误差**



### 反向传播

$y=g(x), z=f(y), x\in\mathbb{R}^m, y\in\mathbb{R}^n  $ 

$\nabla_xz = ({\frac{\partial \mathbf{y}}{\partial \mathbf{x}})^ \mathsf{T} }\nabla_yz\quad, \frac{\partial \mathbf{y}}{\partial \mathbf{x}} 为n\times m Jacobian矩阵$



## 正则化

参数值越小，对英语越光滑的函数，也就是更加简单的函数。因此也不易发生过拟合问题。

通过权衡方差与偏差来实现estimator的正则

### 参数惩罚

正则的目标函数: $ \tilde{J} (\theta ;\mathbf{X},y )=J(\theta;\mathbf{X},y)+\alpha \Omega (\theta), ɑ\in[0, \infty)$

#### $L^2$正则

weight decay / 岭回归

正则项: $\Omega(\theta)=\frac{1}{2}||w||^2_2$![image-20210923154950329](/Users/cxw/Learn/3_Coding/Typora/IMG/image-20210923154950329.png)

>  结果: $\tilde{w} = Q(Λ +ɑ I)^-1Λ Q^Tw^*$
>
> 权重衰减的影响: 沿着轴调节$w^*$, 这个轴是有海塞矩阵的特征向量定义的。
>
> 沿着H特征值非常大的方向，正则的影响很小，而特征值很小的，将衰减为0

在最小二乘的例子中

<img src="/Users/cxw/Learn/3_Coding/Typora/IMG/image-20210923154633207.png" alt="image-20210923154633207" style="zoom: 50%;" />

协方差阵的对角加上了个值

**即， L2正则让算法感知到输入X的高方差，使其缩减那些协方差和输出目标比增加后低的特征的权重**

#### $L^1$正则

Lasso(least absolute shrinkage and selection operator)回归

#### 正则项: $\Omega(\theta)=||w||_1=\Sigma_i|w_i|$

导致一个更加稀疏的结果，即很多参数为0， **实现了一个特征选择**

### 基于约束优化的正规惩罚

建立拉格朗日乘子法，使用KKT条件求解

L1、L2惩罚都可以使用约束实现，如一个L2球面，一个L1菱形

### 数据加强

- 生成fake数据

  - 分类问题

    直接修改 (x, y)中的x

    适合模式识别领域, OCR领域腰小心(6, 9旋转相同)

    适合语音识别领域

- 加入噪声

- dropout

### 多任务学习

提升泛化性能

共享input，对应不同的目标随机变量

### early stopping

### 参数共享

计算机视觉领域的CNN常用

### 稀疏表征

### bagging 其他集成方法

### dropout

通过对某个神经元的weight置0即可

### 对抗学习

## 优化

### 与传统优化的差异

无法跟踪y表现，目标是测试集的error

batch_size设定小批量可以产生一些 正则的效果

极小批量需要更小的学习旅来维持学习的稳定性由于优化梯度存在高方差

