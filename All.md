# ALL

[toc]

## 1 OR

### Linear programming

#### \<Introduction to LP>

---

##### chap1 Introduction

$$
min\quad c'x\\s.t. Ax \ge b
$$
standard form
$$
min\quad c'x\\s.t. Ax=b\\x\ge0
$$
标准化方式：

​		变量 $x = x_j^+ - x_j^-, x_j^+\ge0\quad x_j^-\ge0 $

​		增加松弛$(surplus)$或剩余变量($slack$)，以消除不等式

$convex: f(\lambda x+(1-\lambda)y)\le\lambda f(x)+(1-\lambda)f(y)$ 凹型形状

$max_{i=1,2,…,m f_i(x)} → convex$;  分段线性函数可以来近似凸函数

- ==目标函数存在绝对值时==

  $x_i = x_i^+ - x_i^-,x_i^+,x_i^-\ge0 \Rightarrow x_i=x_i^+ / x_i=x_i^-$  

  then, $|x| = x_i^++x_i^-$

- 解的类型

  - 唯一解
  - 无穷多解
  - 无界
  - 无可行解
  
- Linear Algebra
  
  - k个列向量, $C_k^m$ 种选法，选基
  
- 算法复杂度

  - 矩阵求逆和线性系统求解的复杂度为：$O(n^3)$

##### chap2 The geometry of linear programming 

- Concepts:

  - Ployhedron​

    多面体，是一个$\{x\in\mathfrak{R}|Ax\ge b\}, A:m\times n,b\subset\mathfrak{R}^m$

    - $bounded$ 有界

      存在一个K，使$S\subset\mathfrak{R}^n$中的每个元素$\le$K

  - Hyperplanes

    n维空间下的n-1维  $\{x\in\mathfrak{R}^n|a'x=b\}$, 其中a为超平面的垂直向量

  - Halfspace

    超平面的一侧  $\{x\in\mathfrak{R}^n|a'x\ge b\}$
    
  - Convex Sets

    $S\subset\mathfrak{R}^n,对于S中任意x,y,\lambda\in[0,1],都有\lambda x+(1-\lambda y \in S)$
    
  - Convex Combination 

    $x^1,...,x^k为\mathfrak{R}^n中向量, \lambda_1,...,\lambda_k为和为1的非负数, 向量\sum_{i=1}^k\lambda_ix^i为凸组合$ 

  - Convex Hull

    $x^1,...,x^k的所有凸组合的集合是它的凸包.$
    
  - 定理

    - 凸集的交集还是凸集
    - 多面体都是凸集
    - 一个凸集的有限元素的凸组合仍属于该集合
    - 有限数量的向量的凸包是凸集

- 极点、顶点、基本可行解**[等价]**

  - 极点： 不能由其他两个元素表示为凸组合 ==(该点不在两个点的连线上)==

  - 顶点：存在c使得$c'x\le c'y, 所有y, y\in P,y\ne x.$  

    ==(c为该点的一个方向的向量, P集合在该顶点的一侧/超平面)==

  - 基本可行解

    - 一个向量满足等式约束，即紧约束
    - 集合内的任一个元素都可以有顶点进行凸组合表示
    - 有==n==个线性独立的紧约束成立
      - 基本解：1.所有等式约束成立，2.成立的紧约束中，有n个线性独立
      - 基本可行解：满足所有约束的基本解

- 多面体

  - $P = \{x\in \mathfrak{R}^n|Ax = b, x\ge0\}$,

    $ A:m\times n, m约束数量，n为变量数量$

    

  - 为了获得n个紧约束，我们需要挑选n-m个变量，并设为0，则会使约束$x_i\ge 0$成立n-m个

  - 求解步骤：

    - 选择m个线性无关的列
    - 将其余列队迎的$x_i$ 设为0
    - 求解Ax = b的对应的m个等式的解

  - 邻接基解和邻接基

    邻接基本解：他们中紧约束有n-1个线性独立

    邻接基：只有一个基列不同

- 退化(Degeneracy)

  - 在n维空间内，有超过n个约束为紧约束的基本解称为退化解(比如三个交点连线)
  - 在标准型中：当大于n-m列基变量为0时，该解为退化解

- 极点存在性判别




---

#### \<Optimization in Operations Research>

##### chp3 搜索算法

2. 沿可行改进方向的搜索

   - **方向步长范式**

     $x^{(t+1)} = x^{(t)}+\lambda \Delta x, \lambda\ge0,\lambda$为步长

     - **改进方向**

       对于足够小的$\lambda$, 都有$x^{(t)}+\lambda \Delta x$的目标值优于$x^{(t)}$,则$\Delta x$ 为一个改进方向

   - **可行方向**

     对于足够小的$\lambda$, 都有$x^{(t)}+\lambda \Delta x$的满足所有约束函数,则$\Delta x$ 为一个可行方向
   
   - **确定步长**
   
     最大改进目标值的距离
   
   - 连续搜索算法
   
     - 初始化： $x^{(0)}$
     - 局部最优：不存在可行改进方向
     - 搜索方向：可行改进方向
     - 最佳步长：改进目标值并保持可行的最大步长
     - 前进
   
     ==弱假设下==没有可行改进方向时的停止迭代，可视为局部最优解
   
   - 检验无界：
   
     当某点的可行改进方向可以任意步长改进目标值，则可视为无界
   
3. 可行改进方向的代数条件

   - 改进方向上的梯度条件

     梯度是垂直于该点的目标函数等值面的向量，指向目标函数增长最快的方向

     $\nabla f(x)·\Delta x$ 为当前的增量，进行判断是否为改进方向

   - 将梯度设为改进方向，

     $\Delta x = \pm \nabla f(x)$，最小化取负，最大化取正

   - 紧约束和可行方向

     解x处的搜索方向是否可行取决于该方向上前进任意微小步长是否满足紧约束

   - 线性约束下可行方向的代数条件

     对于大于约束：$\sum_{i=0}^na_jx_j\ge0$  满足时，搜索方向可行

4. 线性目标

   1. 线性目标函数

      $\nabla f(x)·\Delta x$ 为当前的增量

   2. 约束条件和局部最优

   3. 凸集

      可行域内任意两点连线还在可行域内；只要存在更好解，当前解就能有可行改进方向

   4. 连线的代数表示

      两向量的连线是 $x^{(1)}+\lambda (x^{(2)}-x^{(1)})$ 所有点组成

   5. 线性约束可行集的凸性

      模型所以约束为线性时，该模型的可行域时凸集

5. 初始可行解

   人工变量法

   1. 两阶段法

      - 人工问题 

        选择原问题一个解，通过在未满足的约束条件中加入货减去非负人工变量构建第一阶段的人工问题

      - 第一阶段

        给人工变量赋值，得到人工问题的初始可行解，然后执行搜索以最小化人工变量和为目标

        - 第一阶段在 0处停止，可行；全局最小停止，无可行；局部最小停止且$\gt$0, 重新开始

      - 不可行性分析

        人工变量和为0 → 第二阶段；人工变量和$\gt$0则原问题无可行解

      - 第二阶段

        删除第一阶段最优解中的人工变量，得到原问题的初始可行解，常规搜索。

   2. 大M法

##### chp4 线性规划

7. 随机规划

   随机模型包含概率性参数，即随机变量的概率分布已知，需要考虑参数所有可能的取值

##### chp5 单纯形法

1. 标准型

2. 顶点搜索和基本解

   1. 利用紧约束来确定顶点

      标准型中，顶点意味着松弛变量的非负约束为紧约束，松弛变量为0

   2. 相邻的顶点和边

      当两个顶点的紧约束集合只差一个元素时，这两个顶点就是相邻的

   3. 基本解

      标准型中存在非负约束，满足一些非负约束的紧约束的解，叫基本解

      基本解就是令某些变量为0，为0的就是非基变量，等式解出的是基变量 

   4. 基本解的存在性

      仅当选定的基[向量集中最大的线性无关的向量组合]是线性无关时，才有基本解

3. 单纯形法  ==解空间就是$\C_n^m:$ n列变量中，挑选出m个基变量以组成基== 

   - 单纯形方向

     其他非基不变情况下，增加一个非基变量的值，并计算为满足约束，基需要做什么变化，就获得单纯形方向

     $\sum_ja_j\Delta x_j=0$

   - 目标变化

     $\Delta C =\nabla f(x)·\Delta x $

   - 步长和比值最小法则

     - 沿着单纯形方向，$\Delta x$移动，其中有负值，当一个新解中出现第一个0时(保持可行)，就可以确定步长$\lambda$

       $\lambda = min\quad \{ \frac{x_j^{(t)}}{-\Delta x_j}:\Delta x_j<0\}$

     - 优化性单纯形方向$\Delta x$不含负值，则该模型无界

   - 单纯形法算法：

     0. 初始化
     1. 单纯形方向
     2. 最优性判别
     3. 确定步长
     4. 新顶点和基

   - 单纯形表

     单纯形表中，非基变量$x_j$的系数$\bar a_{k,j}$正是增加$x_j$的值所获得的单纯形方向的负值$-\Delta x_k$

4. 退化

   1. 退化解：当一个点的紧约束数量大于确定一个点所需要的约束数量时，发生退化

      基变量的非负约束为紧约束/基变量为0

   2. 零步长

5. 改进单纯形法(**应用于大规模线性规划求解器**)

   - 基矩阵求逆
     - 基：B，基逆：$B^{-1}$ ，基本解：$B^{-1}b$

     - 对某个非基变量$x_j$<u>**进行进基**</u>对应的单纯形方向的$\Delta x_{base} = - B^{-1}a^{(j)}$

   如，对$x^{(3)}$产出的原基$x_1,x_2$的单纯形方向意思为，$\Delta x$ = (1,2,**==1==**)

   - 更新$B^{-1}$

     $new B^{-1} = E(oldB^{-1}), E$ 为更新矩阵

     - 更新矩阵E = [] n维单位矩阵，并将出基列$x_j$更换
       $$
       \begin{bmatrix}
         1&  0&  …&  0&  -\frac{\Delta x_{1st}}{\Delta x_{leave}}&  \dots&  &  0& \\
         0&  1&  …&  0&  -\frac{\Delta x_{2nd}}{\Delta x_{leave}}&  \dots&  &  0& \\
         0&  0&  …&  0&  &  &  &  0& \\
         \vdots&  \vdots&  \ddots&  \vdots&  -\frac{1}{\Delta x_{leave}}&  &  &  & \\
         \vdots&  \vdots&  &  1&  &  &  &  \vdots& \\
         0&  0&  …&  \vdots&  &  \ddots&  &  & \\
         0&  0&  …&  0&  -\frac{\Delta x_{mth}}{\Delta x_{leave}}&  &  &  1&
       \end{bmatrix}
       $$

   - 改进单纯形法中基变量的顺序并非按脚标，而是按出基、进基顺序

   - 成本变化  $Cost:\bar c_j = c_j+\sum_{k\in B}c_k\Delta x_k = c_j - c·B^{-1}·a_j$ 

   - 定价向量(影子价格/) $c·B^{-1}$ 

   - 改进单纯形法

     0. 初始化

     1. 定价：求定价向量 $v = C_B\cdot B^{-1}, 成本减少：\bar c = c_j-v\cdot a^{(j)} $

     2. 最优解： 没有更大的$\bar c\gt 0$ 

     3. 单纯形方向：$B\Delta x = -a^{(p)}, \Delta x = -B^{-1}\cdot a^{(p)}$ 

     4. 步长：最小比值原则

        $\lambda = \frac{x_r^{(t)}}{-\Delta x_r^{(t+1)}} = min\{\frac{x_j^{(t)}}{-\Delta x_j^{(t+1)}}:\Delta x_j^{(t+1)}\lt0\}$

     5. 新基和新顶点

6. 有简单上下限的单纯形法

      - 令非基变量的值等于下限或上限，然后求出出基变量就可以获得基本解

##### chp6 对偶理论和灵敏度分析

1. 对偶模型

   - 对偶变量**[资源边际价格]**

     每个原始问题的主约束对应一个对偶变量(dual variable).该对偶变量的值对应这主约束的RHS系数每增加一个单位，原始问题最优值的变化量

     | 原始模型 | 约束不等号方向 $\le$ | 约束不等号方向 $\ge$ | 约束不等号方向 = |
     | :------: | :------------------: | :------------------: | :--------------: |
     |  $min$   |      $v_i\le 0$      |      $v_i\ge 0$      |   $v_i$无约束    |
     |  $max$   |      $v_i\ge 0$      |      $v_i\le 0$      |   $v_i$无约束    |

     | 原始模型 | 变量类型$\le0$ | 变量类型$\ge0$ | URS    |
     | -------- | -------------- | -------------- | ------ |
     | $min$    | 约束 $\ge$     | 约束 $\le$     | 约束 = |
     | $max$    | 约束 $\le$     | 约束 $\ge$     | 约束 = |

   

   - 生产活动的隐含价格

     $\sum_ia_{i.j}v_i, a_{i,j}$为活动j在约束i左边的系数

   - 对偶约束

     $min: \sum_ia_{i.j}v_i\le c_j  \Leftrightarrow x_j$   

   - **互补松弛定理**

     - 原始解最优时，一个约束不是紧约束，则它对应的对偶变量$v_i=0$

     - 一个原始变量最优值不为0，则对偶价格所在的约束为紧约束

       对偶价格$v_i$使对应对偶约束非紧，则原始变量最优值为0 $x_i=0$

2. 对偶问题和最优解

   1. 目标值的弱对偶性

      $C(primal)_{min} \ge C(dual)_{max}$

      一个问题无界 $\Rightarrow$ 另一个问题无解

   2. 强对偶性
   
      一个问题有最优，则对偶问题和原问题都有最优解，且解值相等
   
   > KKT条件
   >
   > 1. 原始问题有可行解
   >
   > 2. 对偶问题有可行解
   >
   > 3. 互补松弛性是该线性规划问题达到最优的充分必要条件  
   >    $$
   >    (A\bar{x}-b)\cdot \bar{v} = 0,\\(c-\bar{v}A)\cdot\bar{x} = 0,\\ \bar{x},\bar{v}为最优解
   >    $$
   
   3. 分段形式的表示
      $$
      min\quad c^Bx^B + c^Nx^N\\s.t.\qquad Bx^B+Nx^N = 0\\x^B,x^N\ge0\\KKT:(c^B-\bar{v}B)\bar{x}^B = 0\\opt:\bar{x}^B = B^{-1}b, opt-value = c^BB^{-1}b\\DualSolution: \bar{v}B=c^B \Rightarrow \bar{v} = c^BB^{-1}
      $$
   
3. 原始单纯形法的理解

   1. 由可行解开始，并且始终保持原始问题解的可行性
   2. 直接或间接对应有基对偶解，与原始解满足互补松弛条件
   3. 当发生无界或当前基对偶可行解可行时，停止计算

4. 对偶单纯形法

   1. 理解 不断改进对偶搜索策略，最终满足KKT条件

      - 以对偶可行解开始搜索，始终保持对偶解的可行性
      - 每个对偶解对应原问题的基本解
      - 对偶无界/原问题无解/原始解可行时，终止

   2. 优化方向

      $\Delta v \leftarrow \pm r$ ($+:max, -:min$), r为$B^{-1}$和不可行解值$\bar{x}^r$对应的行

      $\bar{v}^{new}b \leftarrow (\bar{v}^{old}+\lambda \Delta v)\cdot b = (\bar{v}^{old}\pm\lambda r)\cdot b = v^{old}\cdot b\pm\lambda r\cdot b$         $max:+\Rightarrow obj\uparrow, min:- \Rightarrow obj\downarrow$

   3. 确定对偶步长，以保持对偶解可行性

      精简成本 $\Delta{\bar{c}}=-\Delta{v}A$  

      $max: \lambda \leftarrow \frac{-\bar{c}_p}{\Delta{\bar{c}_p}} = min\{\frac{-\bar{c}}{\Delta{\bar{c}_j}}:\Delta{\bar{c}_j}\gt0,j非基\}$ 

      $min: \lambda \leftarrow ...$

   4. 算法步骤

      0. 初始化
      1. 最优性 当原始基解满足$\bold{x}^{B(t)}\ge0$,停止计算，原始可行性满足；否则选择原始可行基解r,$x_r^{(t)}\lt0$ 
      2. 对偶单纯形变化方向
      3. 步长大小
      4. 新的解和基

5. 原始-对偶单纯形法搜索

   以对偶可行解起始，一直保留对偶解的可行性；在符合每个对偶的互补松弛条件的有限个原始解中寻求原始可行基解，如果找不到这样的基解，找出对偶优化方向；发现对偶无界(原无界)/当前季基原始可行，终止



##### chp7 线性规划内点法

1. 在可行域内部搜索

   - 内点：严格满足不等式约束的点，在可行域内部

   - 通过目标函数确定移动方向

     - 所有方向都是可行方向
     - 改进最快的方向为 $max: \Delta x =c \\min:\Delta x = -c $

     > 内点法的有效性依赖于，能否在找到最优解之前，一直保持在可行域内部

   - 投影方法处理等式约束

     满足可行移动方向：$A\Delta\bold{x}=0$ 

     给定等式下的**移动向量**($\pm c$)d的投影，是一个满足约束且最小化投影分量和d分量平方差的方向

     $\Delta \bold{x} = Pd, P = (I-A^T(AA^T)^{-1}A)$  ,**这个方向是一个改进方向** [$c\Delta\bold{x}>0$]

2. 尺度变换(Scaling)

   通过修改决策变量单元，<u>让解所有分量与边界保持合适距离</u>，典型：仿射(affine)   

   1. 仿射尺度变换

      当前内点解$x^{(t)}$, 仿射 ：$y_1 = \frac{x_1}{x^{(t)}_1}$

   2. 对角阵建立
      $$
      当前解：x_t = (x_1^{(t)},\dots,x_n^{(t)})\\X_t=\begin{bmatrix}
      x_1^{(t)}&0&\cdots&0\\
      0&x_2^{(t)}&\ddots&\vdots\\
      \vdots&\ddots&\ddots&0\\
      0&\cdots&0&x_n^{(t)}\\
      \end{bmatrix},
      \qquad X_t^{-1} = \begin{bmatrix}
      \frac{1}{x_1^{(t)}}&0&\cdots&0\\
      0&\frac{1}{x_2^{(t)}}&\ddots&\vdots\\
      \vdots&\ddots&\ddots&0\\
      0&\cdots&0&\frac{1}{x_n^{(t)}}\\
      \end{bmatrix}
      $$
      

      $y = X_t^{-1}\bold{x}\quad /\quad y_j = \frac{x_j}{x_j^{(t)}} $

   3. 仿射变换后的标准型
      $$
      min\quad cX_t\cdot y\\s.t. AX_ty=b\\y\ge0
      $$

3. 仿射尺度变换搜索

   仿射后的当前解的每个分量$y^{(t)}=1$， 变换后的点与非负约束距离相等

   1. 移动方向

      $max:\Delta{y}=P_tc^{(t)} \\\Delta{x} = X_t\Delta{y}=X_tP_tc^{(t)}$

   2. 步长

      $\lambda = \frac{1}{||\Delta\bold{x}X_t^{-1}||} = \frac{1}{||\Delta{y}||}$ 

   3. 搜索停止

      求出每一步最优值的边界，让我们得知当前解的改进空间

   4. 算法步骤

      0. 初始化
      1. 最优化
      2. 移动方向
      3. s
      4. 步长
      5. 前进

4. 对数障碍法

   1. 障碍目标函数

   $x_j \rightarrow 0,有\ln(x_j)\rightarrow-\infin$，障碍函数包括对数，使$x_j$远离边界

   $max\rightarrow max\quad\sum_jc_jx_j+\mu\sum_j\ln(x_j)\\min\rightarrow max\quad\sum_jc_jx_j-\mu\sum_j\ln(x_j), \mu\ge0$ 

   2. 梯度方向问题

      可以给出梯度方向，但不适合用来处理非线性目标函数

   3. 牛顿步

      $f(x+\lambda\Delta{x})\approx {f(x^{(t)})+\lambda\sum_j\frac{\partial f}{\partial x_j}\Delta x_j+\frac{\lambda^2}{2}\sum_j\frac{\partial^2 f}{\partial x_j \partial x_k}\Delta x_j}$

      其中, $\frac{\partial f}{\partial x_j} = c_j\pm \frac{\mu}{x_j}\\\frac{\partial^2 f}{\partial x_j \partial x_k} = \mp\frac{\mu}{(x_j)^2}, if\quad j=k$

      移动方向:$max:\Delta\bold{x}=+(\pm) X_tp_t[{c_1^{(t)\pm \mu}\\c_n^{(t)\pm \mu}}]$

      移动步长：$\lambda = min\{\frac{1}{\mu},0.9\lambda_{max}\}, \lambda_{max}=min\{\frac{x_j^{(t)}}{-\Delta x_j^{(t+1)}}:\Delta x_j^{(t+1)}\le0$

      障碍因子$\mu\gt0$, 较大时，对接近边界的内点阻碍效果显著，较小时，鼓励搜索过程接近边界
      
      **策略**：障碍算法以较大的障碍因子$\mu\gt0$出发，随着搜索过程进行，缓慢减小至0

   4. 牛顿步障碍算法
      0. 初始化：可行内点$x^{(0)}\gt0$和较大的障碍因子$\mu$
      1. 移动方向：向**仿射尺度变换空间投影**：$\Delta{x^{(t+1)}}\leftarrow\pm{X_tP_tc'^{(t)}}$
      2. 步长：$\lambda\leftarrow min\{\frac{1}{\mu},0.9\lambda_{max}\}$ 
      3. 前进
      4. **内循环**：若改进较大，可以看出障碍因子为$\mu$ 的情况下，$\bold{x}^{(t+1)}$距最优值较远，返回第一步，继续迭代
      5. **外循环**：若障碍因子$\mu$接近0，停止搜索，当前解最优/近似最优；否则减小障碍因子，返回第一步

5. 原始对偶内点法

   1. KKT最优性条件

      原问题可行+对偶问题可行+互补松弛条件$\quad\bar{x}_j\bar{w}_j=0, w$为对偶问题补成等式的非负松弛变量

   2. 策略

      原始对偶内点法始终保持原始问题和对偶问题的解在每次迭代中严格可行，并在搜索过程中系统性减小**互补松弛性的违背程度**

   3. 可行移动方向

      需满足: $A\Delta{x}=0\\A^T\Delta{v}+\Delta{w}=0$

   4. 互补松弛性的管理

      对偶间隙=$c\bar{x}-b\bar{v}=\sum_j\bar{x}_j\bar{w}_j=$总互补松弛违背程度

      由于$\bar{x},\bar{w}$严格可行，每一项$\bar{x}_j\bar{w}_j\gt0$ 

      添加对互补松弛的违背程度减小目标

      $(\bar{x}_j+\Delta{x_j})(\bar{w}_j+\Delta{w_j}))=\mu, all\quad j$,为了使求解方向的条件近似线性，将不考虑$\Delta{x_j}\Delta{w_j}$项

   5. 步长

      $\lambda\leftarrow \delta min\{\lambda_P,\lambda_D\},\delta为障碍因子\\\lambda_P=min\{-\frac{\bar{x}_j}{\Delta\bar{x}_j}:\bar{x}_j\lt0\}\\\lambda_D=min\{-\frac{\bar{w}_j}{\Delta\bar{w}_j}:\bar{w}_j\lt0\}\\$

   6. 算法步骤

      0. 初始化：

         - 找到一个严格可行的原问题解$x^{(0)}$
         - 一个严格可行的对偶问题的解($v^{(0)},w^{(0)}$)
         - 一个目标减小因子$0\lt\rho\lt1$
         - 计算对偶间隙$g_0\leftarrow{cx^{(0)}}-bv^{(0)}$
         - 让$\mu_0 \leftarrow g_0/n,n$为x的维度

      1. 最优化：对偶间隙足够接近0时，否则$\mu_{t+1} \leftarrow \rho\cdot{\mu_t}$

      2. 移动方向：求解下列方程组$\\A\Delta{x}=0\\A^T\Delta{v}+\Delta{w}=0\\x_j^{(t)}\Delta{w^{(t+1)}_j}+w_j^{(t)}\Delta{x^{(t+1)}_j}=\mu_{t+1}-x^{(t)}_jw^{(t)}_j$  

         **矩阵式求解：** $\Delta{v}^{(t+1)}\leftarrow-[AW_{t}^{-1}X_tA^T]^{-1}(\mu_{t+1}1-X_tW_T1)\\\Delta w^{(t+1)}\leftarrow -A^T\Delta{v}^{(t+1)}\\\Delta{x}^{(t+1)}\leftarrow-W^{-1}_tX_t\Delta w^{(t+1)}+W^{-1}_t(\mu_{t+1}1-X_tW_T1)$  

      3. 步长：$\lambda\leftarrow \delta min\{\lambda_P,\lambda_D\}$

      4. 前进，返回第1步

6. 线性规划搜索算法的复杂性

   - 标准型的长度定义

     $L\triangleq n\cdot m +\sum_j[\log(|c_j|+1)]+\sum_{ij}[\log(|a_{ij}|+1)]+\sum_j[\log(|b_i|+1)]$ 

     $n, m, c, A, b$分别为变量数、约束数、目标系数、约束矩阵、右端项

   - 单纯形法复杂度

     指数级复杂度，是L的函数

     遍历所有顶点

   - 内点法复杂度 

     涉及投影方法的内点法复杂度:$O(\sqrt{n}L)$次迭代，实现**多项式时间**的实现

##### chp8 目标规划

1. 有效点和有效边界

   - 有效点(帕累托点和非支配点)

     有效点不能被其他可行解完全占优

   - 有效边界

2. 抢占式和加权目标

   - 抢占式优化
   - 加权求和

3. 目标规划

   强调目标的实现程度，而非数量上的极大极小

   硬约束：可行性；软约束：应尽量满足但允许违背：首选方案

   (目标函数+偏差变量$\ge$目标水平)

   目标函数：最小化加权偏差量

##### chp9 最短路和离散动态规划

1. 最短路模型

   1. 基本概念

      - 图：在网络中对运动、流、相邻元素所建立的模型
      - 节点：表示网络中的实体、交点和转移点
      - 弧：节点之间的有向连线
      - 边：节点之间的无向连线
      
   2. 路(path)
   
      两个特定节点的一序列的弧或者边，弧或者边要去前面的弧、边有一个共同点，且没有节点会被一条路重复经过
   
2. 动态规划求解

   1. 符号

      $v[k]\triangleq$ 从起点到节点k的最短路长度($+\infin$则无可行路径)

      $x_{i,j}[k]\triangleq \left\{\begin{aligned} 1,若弧/边(i,j)是起点到终点k最短路的一部分\\0,否则\end{aligned}\right.$

      $v[k,l]\triangleq$ 从节点k到节点l的最短路长度($+\infin$则无可行路径)

      $x_{i,j}[k,l]\triangleq \left\{\begin{aligned} 1,若弧/边(i,j)是节点k到终点l最短路的一部分\\0,否则\end{aligned}\right.$

   2. 最优性定理

      在一个没有**负权环路**的图中，最优路径一定包含着最优子路径

   3. **动态规划的函数方程**

      一对多：

      $v[s] = 0\\v[k] = min\{v[i]+c_{i,k}:存在弧(i,k)\} \forall{k\neq{s}}，要考虑所有与k直接相连的所有点i$

      多对多

      $v[k,k] = 0,\forall{k}\\v[k,l] = min\{\{v[k,i]+v[i,l]\}+c_{k,l}:i\neq l\} \forall{k\neq{l}}，要考虑k,l之间的任意一个中介点i$

3. 一对多：[Bellman-Ford算法](/Users/cxw/Learn/2_SIGS/ALi/Coding/OR/ShortestPath/Bellman_Ford.py)

   算法的核心在于重复评价。每轮搜索都必须要利用上一轮的结果计算函数方程的$v[k]$，直到结果不再变化

   1. 算法

      $v^{(t)}[k]\triangleq $ 第t次循环得到的$v[k]$值

      $d[k] \triangleq$ 从s到k的已知最优路径上的k的前向节点

      算法步骤：

      0. 初始化

         $v^{(0)}[k]= \left\{ \begin{aligned}0,k=s\\+\infin,other \end{aligned}\right.$  

      1. 评价

         $v^{(t)}[k] = min\{v^{(t-1)}[i]+c_{i,k}:存在弧(i,k)\}$ 

         若$v^{(t)}[k]<v^{(t-1)}[k])$ ,则同时设定 $d[k] \leftarrow 使得v^{(t)}[k]达到最小值的相邻节点i$ 

      2. 停止

         若对所有k，有$v^{(t)}[k]=v^{(t-1)}[k]$，或t达到了图中的节点数量

      3. 跳转 

         $v^{(t)}[k]$仍在变化，则t小于节点数，$t\leftarrow t+1$,跳转步骤1

   2. 算法复杂度

       **$O(n^3)$**

       n个节点：n次循环$O(n)$; 循环内，考虑n个节点的相邻流入节点(最差是n个)：$O(n^2)$

   3. 负权环路下的Bellman-Ford算法

      到t=节点数后，仍变化的节点在负权环路之中。

4. 多对多：[Floyd算法](/Users/cxw/Learn/2_SIGS/ALi/Coding/OR/ShortestPath/Floyd.py)

   $v^{(t)}[k,l]\triangleq$ 中介节点的数量小于等于t时的从k到l的最短路长度

   $d[k,l]\triangleq$ 目前从k到l的最短路中l的前一个节点

   1. 无负权环路的*FLoyd*算法

      0. 初始化

         - 对于所有的弧和边：

           $v^{(0)}[k,l]\leftarrow c_{k,l}$ 

           $d[k,l]\leftarrow k$

         - 对于不存在弧和边的节点k,l

           $v^{(0)}[k,l]=\left\{\begin{aligned} {0, k=l\\+\infin,other }\end{aligned}\right.$

      1. 评价

         对每个$k,l\neq{t}$更新：

         $v^{(t)}[k,l]\leftarrow min{\{v^{(t-1)}[k,l],v^{(t-1)}[k,t]+v^{(t-1)}[t,l]\}}$

         若被更新优化，则设定$d[k,l]=d[t,l]$

      2. 停止

         对于某个节点$k$有$v^{(t)}[k,k]<0$,或t达到图中节点数，停止计算

         若存在负值，则图中包含负权环路

      3. 跳转

         如果对于所有节点$v^{(t)}[k,k]\ge0$，而且t小于节点数，那么$t\leftarrow t+1$,跳转步骤1

   2. 时间复杂度

      $O(n^3)$,  经过n次循环，每个循环内检查$O(n^2)$对节点

5. 无负权一对多：[Dijkstra算法](/Users/cxw/Learn/2_SIGS/ALi/Coding/OR/ShortestPath/Dijkstra.py)

   1. Dijkstra 

      某个节点被永久标签后，其$v[p]$和$d[p]$将不再变化

      只适用于一对多，且每条路上的成本都是非负

      0. 初始化

         记s为起点

         $v[i] = \left\{ \begin{aligned}0, i=s\\+\infin,other \end{aligned}\right.$

         所有点记为临时标签点，$p\leftarrow s$作为下一步永久标签点

      1. 处理

         节点p记为永久标签，对所有从p出发到一个永久标签节点的弧/边$(p,j)$，更新：

         ​	$v[i]\leftarrow min\{ v[i], v[p]+c_{p,i}\}$

         ​    设定 $d[k]\leftarrow p$

      2. 停止

         如果不存在临时标签节点，算法终止

      3. 跳转

         选择当前$v[i]$最小的临时标签节点p作为下一次的永久标签，

         ​    $v[p] = min\{v[i]:节点i为临时标签\}$

         跳转步骤1

   **Dijkstra算法只使用永久标签的节点来计算从起点s到i的最短路长度$v[i]$** 

   选择最小$v[i]$作为下一个永久标签的原理：没有比最小的这个p 更短的路

   2. 复杂度

      $O(n^2)$ n个节点：n次循环$O(n)$；每次循环包括检查临时标签点$O(n)$

6. 一对多无环图最短路问题

   1. 无环图

      图中只有有向弧没有边，没有环

      无环图的判断：

      - 仅当它的所有节点在标号后可以保证每条弧(i, j)都有 i < j.
      - 标号方法：深度优先，出弧指向已被标记则按从大到小的顺序给该节点标号【回溯】

   2. 最短路算法

      0. 初始化

         弧的标号，要求满足每条弧$(i,j), i<j$

         $v[s] \leftarrow 0$ 

      1. 停止

         当所有$v[k]$不再发生变化后，停止计算;否则p为未处理的最小编号的点

      2. 处理

         若节点p不存在入弧，$v[p]\leftarrow\infin$ ；否则考察p点的入弧：

         $v[p]\leftarrow min\{v[i]+c_{i,p}:\exists arc(i,p)\}$ 

         $d[p]\leftarrow$ 达到最小值的节点i的数字

         跳转步骤1

   3. 复杂度

      考虑了所有的弧，复杂度$O(弧的数量)$ 

   4. **最长路问题**

      稍作改变，将$min \rightarrow max, +\infin \rightarrow -\infin$ ,如无环图最短路算法

7. 离散动态规划

   1. 序贯决策问题

      动态规划对应的有向图的节点是中间过程的状态，弧对应着决策

   2. 拥有阶段(stage)和状态(state)的动态规划模型

      - 阶段：描述了需要做的决策序列
      - 状态：描述了决策需要考虑的那些条件

   3. 复杂度

      $n\triangleq$ 阶段数量，$m\triangleq$每个阶段最大状态数量

##### chp11 离散优化模型

1. 全/无条件下的整数规划建模

   $x_j = 0\ or\ u_j \Rightarrow x_j = u_jy_j, y_j=0\ or \ 1$

2. 固定成本

   成本函数：$\theta(x) = \left \{\begin{aligned}f+cx,x>0\\0, otherwise. \end{aligned} \right.$

   可以将固定成本单独列为一个01变量，系数为固定成本

   **开关约束**：$x_j \le u_jy_j, u_j$为$x_j$的上界

3. 互斥选择

   $\sum{x_j}\le1$

4. 依赖关系建模

   i 对 j 的依赖：$x_j\le{x_i}$

---

###### ==线性化方法==

1. $max, min$

   $z = min\{x,y,5\} \Rightarrow x\ge z, y\ge z, 3\ge z\\x \le z-M(1-u_1),y\le z-M(1-u_2),3\le z-M(1-u_3)\\u_1+u_2+u_3 \ge 1\\u_1,u_2,u_3\in \{0,1\}$ 

2. obj中绝对值

   $min \sum_ici|x_i|$

   1. $y_i=|x_i|,y_i\ge x_i,y_i\ge -x_i$
   2. $\forall x,\exists u,v\gt0, x= u-v,|x|=u+v, 其中u = \frac{|x|+x}{2},v= \frac{|x|-x}{2}$ 

3. $max(min)/ min(max)$

   $max(min_{k\in K}\sum_ic_{ki}x_i)$

   $\Rightarrow max\ z\\z\le\sum_ic_{ki}x_i,\forall k\in K$

4. $max(max)/min(min)$

   $max(max_{k\in K}\sum_ic_{ki}x_i)$

   $\Rightarrow max\ z \\\sum_ic_{ki}x_i\ge z-M(1-y_k),\forall k \in K\\\sum_k y_k \ge 1, y_k \in \{0, 1\}$

5. 分式

   取出分母，将乘积式代替

6. 逻辑或

   $\sum_ja_{1j}x\le b_1\ or\ \sum_ja_{2j}x\le b_2$ 两个约束至少成立一个

   $\sum_ja_{ij}x\le b_i+M(1-y_i), i=1,2\\\sum_i y_i\ge1,y_i\in \{0,1\}$
   
7. 乘积式
   $$
   y = x_1+x_2, x1, x_2 \in \{0,1\}\\
   y\le x_1\\y\le x_2\\ y \ge x_1+x_2-1\\y\in \{0, 1\}
   $$

8. 


---

###### ==整数规划建模技术==

1. 两个条件至少一个满足$a'x\ge b,\ c'x\ge d$

   $a'x\ge{yb},\ c'x\ge(1-y)d, y\in\{0,1\}$ 

   m个条件至少k个满足:

   $a_i'x\ge b_iy_i, \ \sum_{i=1}^m y_i \ge k$ 

2. x取定值

   则给出定值集合对应的二元变量来表示x

---

##### chp12 离散优化求解方法

1. 枚举法

2. 松弛$(relaxation)$

   - 约束条件的松弛

     要求目标函数相同，且原问题$P$的可行解都在松弛问题$\bar{P}$中

   - 线性规划松弛[连续松弛]

     将离散变量看成连续变量

   - 修改目标函数的松弛模型

     两个优化问题$R, P$, $Solution_P\subset Solution_R$; $Obj_R(\bold{x})\ better\ than Obj_P(\bold{x})$ ,则$R$是$P$的松弛问题

   - 松弛的效果
     - 证明不可行
     - 获得原问题的界限
     - 获取最优解
     - 用来取整数解
     
   - 有效的松弛

     - 原问题可行解一定是松弛问题可行解
     - 松弛问题最优解一定优于或等于原问题最优解

3. 分支定界搜索

   - 部分解

     一个部分解：具有一些固定不变的决策变量，和其他自由的/不确定的决策变量

     部分解的完全形式，是符合部分解中全部固定分量要求的可能的完整解

   - 树搜索

     - 搜索根解$root:x^{(0)} =(\#,\dots,\#)$ ,所有变量都是自由变量

     - 分支：一个部分解不能被终止，则要分支。

       ​			分支方式：固定一个自由变量，分为两个子部分

     - 停止条件：每个部分解都已被分支或终止

     - 搜索方式：深度优先

   - 最佳解$(incumbent\ solution)$

     $\hat{\bold{x}}$, 对应目标函数值$\hat{v}$ 

   - **基于线性规划的分支定界法(0-1整数规划)**

     0. 求初始解

        让所有离散变量均自由

        $max: \hat{v}\leftarrow{-\infin};min: \hat{v}\leftarrow{+\infin}$ 

     1. 停止

        存在活跃的部分解，选择一个作为$\bold{x}^{(t)}$，转步骤2

        否则，停止，已存在最佳解，则该最佳解为最优解；不存在最佳解，则该问题不可行

     2. 松弛

        求解与$\bold{x}^{(t)}$对应的线性规划松弛模型

     3. 通过不可行终止

        若该线性规划松弛模型证明不可行，则部分解$\bold{x}^{(t)}$没有任何可行的完全形式，终止$\bold{x}^{(t)}$，跳转步骤1

     4. 通过定界终止

        当$max$线性规划松弛模型最优解$\widetilde{v}\le\hat{v}$ ,则部分解$\bold{x}^{(t)}$的最优可行的完全形式不能使最佳解更优，终止$\bold{x}^{(t)}$，跳转步骤1

     5. 通过求解终止

        线性规划松弛模型的最优解$\widetilde{x}^{(t)}$满足约束，则该$\widetilde{x}^{(t)}$ 是部分解$\bold{x}^{(t)}$的最优可行的完全形式

        $\hat{x}^{(t)}\leftarrow \widetilde{x}^{(t)} \\\hat{v}\leftarrow\widetilde{v}$

        终止$\bold{x}^{(t)}$，跳转步骤1

     6. 分支

        选择线性规划松弛模型最优解中的某些为分数的自由二元变量$x_p$,将$x_p$分别固定为0, 1。跳回步骤1

        (倾向于选择一个最接近整数值的分量来进行分支)

   - **分支定界法的改良**

     - 整数化松弛最优解，提供一个新的最佳解

     - 利用母节点界限终止

       ​	子节点$child$, 母节点$parent$

       ​	母节点的松弛最优值是子节点目标值的上界($max$)/下界($min$) 

       当新的最佳解出现时，某个部分解的母节点界限比最佳解差时，可以直接终止该部分解

     - 提前停止，仅获取最新的最佳解

       - 误差界限

         母节点的界限与当前最优解的差值是可能的最大gap = $\frac{可能的最优值-知道的最优值}{知道的最优值}$ 

     - 分支顺序

       - 深度优先

         选择具有最优固定分量的活跃部分解( 树搜索中最深的节点)

       - 最优优先

         每个循环阶段，选择具有最优母节点界限的活跃部分解

       - 深度向前最优回溯

         分支后选择最深的活跃部分解，终止一个节点后，选择最优母节点界限的活跃部分解

       最深或最优母节点界限不唯一时，**最近子节点**原则选择具有最后固定的变量值，离母节点线性规划松弛解相应分量取值最近的子节点部分解。

     - 利用母节点线性规划最优解，作为子节点的初始解

4. 分支切割法

   1. 有效不等式

      线性不等式对所有( 整数)可行解都成立，则为该离散优化模型有效不等式

   2. 分支切割搜索

      分支切割算法，在分支部分解之前，用新的有效不等式来加强松弛模型的有效性，应切割掉松弛模型最优解

      0. 求初始解

      1. 停止

      2. 松弛

      3. 通过不可行终止

      4. 通过定界终止

      5. 通过求解终止

      6. 有效不等式

         试着找出一个完整整数规划模型的有效不等式，使当前松弛最优解$\widetilde{x}^{(t)}$不满足这个不等式，加入至完整模型，返回步骤2

      7. 分支

5. 有效不等式组

   1. Gomory割平面(纯整数规划模型)

      小数部分 $\phi(q)\triangleq q-\lfloor{q}\rfloor$ 

      **Gomory小数割平面方程**，任意行$k$, 令$f_{k0}$表示$\phi(\bar{b}_k),\ f_{kj}$表示$\phi(\bar{a}_{kj})$ :

      $\begin{aligned} \sum_j{f_{kj}x_j}\ge{f_{k0}}  \end{aligned}$  

      >  **[Gemory割的证明]**
      > $$
      > x_k = \bar{b}_k - \sum_ja_{kj}x_j\\其中，\bar{b}_k = \lfloor{b_k}\rfloor+f_{k0},\  a_{kj} = \lfloor{a_{kj}}\rfloor + f_{kj}\\f_{k0}-\sum_jf_{kj}x_j = x_k-\lfloor{b_k}\rfloor - \sum_j\lfloor{a_{kj}}\rfloor{x_j}\\RHS 为纯整数, 0\lt f_{kj}\lt1,x_j\ge0,\ \Rightarrow\sum_jf_{kj}x_j\gt0\\0\lt f_{k0}\lt 1\\\Rightarrow f_{k0}-\sum_jf_{kj}x_j<1\ 且为整数
      > \\\therefore\ \bold{f_{k0} -\sum_jf_{kj}x_j \le0}
      > $$

      **更强有力的有效割平面**
      $$
      \ \sum_{f_{kj}\le{f_{k0}}}f_{kj}x_j + \sum_{f_{k0}\gt{f_0}}\frac{f_{k0}}{1-f_{k0}}{(1-f_{kj})}x_j\ge{f_{k0}}
      $$

      > 证明：
      > $$
      > x_k = \bar{b}_k - \sum_ja_{kj}x_j\\x_k +\sum_{f_{kj}\le{f_{k0}}}(\lfloor\bar{a}_{kj}\rfloor+f_{kj})x_j+\sum_{f_{kj}\gt{f_{k0}}}(\color{red}{\lceil\bar{a}_{kj}\rceil-(1-f_{kj})})x_j =\lfloor\bar{b}_{k}\rfloor+f_{k0}
      > \\x_k+\sum_{f_{kj}\le{f_{k0}}}\lfloor\bar{a}_{kj}\rfloor{x_j}+\sum_{f_{kj}\gt{f_{k0}}}\lceil\bar{a}_{kj}\rceil{x_j}-\lfloor\bar{b}_{k}\rfloor
      > \\= f_{k0}-\sum_{f_{kj}\le{f_{k0}}}f_{kj}x_j+\sum_{f_{kj}\gt{f_{k0}}}(1-f_{kj})x_j
      > $$
      > 左边为整数,则右边也为整数
      >
      > - $LHS\le0$
      >   $$
      >   \sum_{f_{kj}\le{f_{k0}}}f_{kj}x_j-\sum_{f_{kj}\gt{f_{k0}}}(1-f_{kj})x_j\ge f_{k0}
      >   $$
      >   
      >
      > - $LHS\ge1$
      >   $$
      >   -\sum_{f_{kj}\le{f_{k0}}}f_{kj}x_j+\sum_{f_{kj}\gt{f_{k0}}}(1-f_{kj})x_j\ge 1-f_{k0}
      >   $$
      >   两边同乘$\frac{f_{k0}}{(1-f_{k0})}$
      >   $$
      >   \ -\sum_{f_{kj}\le{f_{k0}}}f_{kj}\frac{f_{k0}}{1-f_{k0}}x_j + \sum_{f_{k0}\gt{f_0}}\frac{f_{k0}}{1-f_{k0}}{(1-f_{kj})}x_j\ge{f_{k0}}
      >   $$
      >
      > 两约束右手边同样大于一个数，可以根据$f_{kj}和f_{k0}$ 的关系来选择一个更大的系数产生更有力的割。

   2. Gomory割平面[MIP]

      其中是连续变量的，$a_{kj}$进行保留

   3. 特定模型的有效不等式组

6. 割平面理论

   1. 概念

      - 凸包($convex\ hull$)

        最小包含所有可行解的凸集

      - 凸包给出了最强有效不等式的理想参考点

      - 有效不等式

        - 面：凸包的面是任何满足一些有效不等式中等号的点构成的子集
        - 侧面：具有最大维度的面，其维度比凸包的维度少

   2. 要求向量仿射独立，仅要求向量的差值独立

      IP/MIP中，当且仅当存在k+1个仿射独立且满足不等式等号部分的整数可行解时，模型的一个有效不等式会对应到凸包的一个维度为k 的面。

      特别的，当存在n个，n是凸包维度，有效不等式就是侧面的对应有效不等式



##### chp13 ==大规模优化方法==

###### **列生成(Column generation)**

用于求解每个<u>决策方案对应</u>整体规划模型中<u>约束矩阵的一列</u>的组合优化问题

基于当前生成的列的子集，限制主问题进行优化求解，其余候选方案只有当列生成子问题判别可以改善限制主问题当前最优解时，才会被选择进入该子集
$$
max(min) \ \sum_{j\in J}c_jx_j\\
ELP(J)\qquad\ s.t. \ \sum_{j\in{J}}a^{(j)}x_j\le{b}\\
x_j\ge0,\forall\ j \in J
$$
$J_{all}$为所有可能的列构成的集合，$J\subseteq J_{all}$ 表示当前限制主问题中包含的列的子集

- 算法步骤

  0. 初始化

     设迭代次数$l$为0，选择一个子集$J_0\subseteq{J_{all}}$，使得对应的限制主问题$ELP(J_0)$存在可行解

  1. 求解主问题

     求解限制主问题$ELP(J_{l})$, 找到相应的最优解$\mathbf{x}^{(l)}$和对偶问题  最优解$\mathbf{v}^{(l)}$

  2. 列生成子问题

     考虑一个新的决策变量对应的列$\mathbf{a}^{(g)}$, 若该列满足所有列相关的复杂度约束条件$g\in{J_{all}}$, 并且对于目标函数最大化问题，检验数(影子价格) $\bar{c}_g\triangleq c_g-a^{(g)}v^{(l)}\gt0$则选择该列加入约束矩阵

     [此处，检验数>0(<0), 意味着该列进基能够改善当前解]

  3. 判断算法终止条件

     没有新列则终止算法，此时解$\mathbf{x}^{(l)}$为最优货近似最优解。

     否则，更新限制主问题约束矩阵：$J_{l+1}\leftarrow{J_l\cup{g}},\ ;\leftarrow l+1$, 跳转第一步

[列生成实现示例代码(PythonCallGurobi)](/Users/cxw/Learn/2_SIGS/ALi/Coding/OR/ScaleOptimization/ColumnGeneration.py)

> ==遇到的问题==：Gurobi不提供IP/MIP的对偶解，所以当手写对偶问题进行求解对偶解/Reduce cost时，**不能对对偶变量作整数限制**，否则将得不到对应的对偶最优解。

###### **分支定价(branch and price)**

本质上是 $branch\ and\ bound \ \&\ Column\ generation$ 

- 算法步骤

  0. 初始化

     - 求解原问题对应的不含整数约束的子问题，该解作为分支定界树的根节点;

       放入当前**需要分支的节点列表$(active\ partial\ solution)$** ;

     - 迭代次数$l \leftarrow 0$;

     - 若求得整数可行解，选择其中最优的$\hat{\mathbf{x}}$ 当作最佳解，目标值记为最佳值$\hat{v}$;

       否则, $max(min):\hat{v}\leftarrow -\infin(+\infin)$ .

  1. 算法停止

     - 分支节点列表包含节点元素：按照策略选定一个节点，记$\mathbf{x}^{(l)}$，跳转第2步;
     - 分支节点列表不包含节点元素： 算法停止，存在最佳解则为最优解，不存在则模型不可行.

  2. 线性松弛

     求解$\mathbf{x}^{(l)}$对应节点子问题线性松弛问题

  3. 更新当前最优解

     如果松弛问题最优解$\widetilde{\mathbf{x}}$满足0-1整数约束，并且得到了更优的目标值$\widetilde{v}$，更新

     $\hat{\mathbf{x}}\leftarrow\widetilde{\mathbf{x}},\ \hat{v}\leftarrow\widetilde{v}$ 

  4. 列生成

     获得第二步限制主问题得到的最优对偶变量$\widetilde{v}^{(l)}$代入到相应的列生成子问题中

     - $max(min):$选择为+(-)的检验数的列，加入当前子问题的限制主问题中，返回第2步；
     - 否则当前解是该节点子问题最优解

  5. 基于最优解可行性的分支终止条件判断

     当前节点的解不满足该分支下的**整数约束**，终止分支，跳转第1步

  6. 基于目标函数值上下界的分支终止条件判断

     当前节点的松弛最优解小于最佳解，终止分支，跳转第1步

  7. 基于线性松弛问题整数解的分支终止条件判断

     该线性松弛问题最优解$\widetilde{\mathbf{x}}^{(l)}$满足整数约束，则该解为该分支下的最优整数解，终止该分支，跳转第1步

  8. 分支

     按照一定策略，选择不符合0-1整数条件的变量$x_p$，构造两个子部分约束，得到两个规划问题，加入待分支列表，跳转第1步

**==TODO：代码实现==** 

Gurobi的数据结构和算法无法在求解中加入一列变量； **SCIP**支持

###### **拉格朗日松弛(Lagrangian Relaxation)**

**本质**：$ma x_{\lambda}\ min_x\ L(x,\lambda)$, 对$x, \lambda$进行迭代更新

1. 拉格朗日松弛

   放松模型中的部分线性约束, 利用$Lagrangian$乘子在目标函数上增加惩罚项

   $\qquad\qquad\dots + v_i(b_i-\sum_ja_{i,j}x_j),\quad v_i$为约束条件i对应的$Lagrangian$乘子

   - $max(min):\ \ \sum_ja_{ijx_j}\ge{b_i}, v_i\le0(\ge0)$
   - $max(min):\ \ \sum_ja_{ijx_j}\le{b_i}, v_i\ge0(\le0)$
   - $\sum_ja_{ijx_j}={b_i}, v_i$符号无限制 

2. 界和最优解

   添加的惩罚项只会改进目标值

   $max(min):Larangian$松弛问题的最优目标函数值是原问题的下界(上界)

   **原理：当$Lagrangian$松弛问题最优解能够满足每个被松弛的不等式约束的互补松弛条件，$v_i\times(b-\sum_ja_{ij}x_j)=0$ ,则该解是原问题的最优解**

   > 证明：
   >
   > 互补松弛条件保证了每个被松弛的约束咋目标函数中的对应的对偶项等于0
   
3. 拉格朗日对偶问题
   $$
   min\ \ \mathbf{cx}
   \\(P)\ \ s.t.\ \ \mathbf{Rx}\ge\mathbf{r}
   \\ x\in T \triangleq \{\mathbf{x}\ge0:\mathbf{Hx}\ge\mathbf{h}, x_j\in N,j\in J\}
   $$
   拉格朗日松弛问题
   $$
   min\ \  \mathbf{cx+v(r-Rx)}
   \\(P_v)\ \ s.t. \mathbf{x}\in\mathbf{T}
   $$
   求解下面的问题来获得最优的拉格朗日乘子
   $$
   \mathbf{max}\ \ v(P_v)
   \\(D_L)\ \ s.t. \ \ \mathbf{v}\ge0
   $$
   本质上，即让松弛问题和原问题具有相同的界限

   - 对于$(P), (P_{\bar{\mathrm{v}}}), (D_L)$ ，一定有$v(P)\ge v(D_L)$.设乘子$\hat{\mathbf{v}}\ge0$对应的松弛问题$(P_{\hat{\mathbf{v}}})$的最优解为$\hat{\mathbf{x}}$, 当满足$\mathbf{R\hat{x}}\ge\mathbf{r},\mathbf{v(r-R\hat{x})}=0$, 则 $\hat{\mathbf{x}}$ 也是原问题(P)的最优解, 且$v(P)=v(D_L)$.

   - 求解拉格朗日对偶问题：是求**[保留约束条件下的凸包]**下的最优解；

     求解拉格朗日线性松弛问题：是松弛线性后的最优解

   - 松弛问题最优函数值$v(P_v)$是关于v的分段线性**凹函数**

4. 求解拉格朗日界的次梯度优化算法(最速下降的推广)

   最优乘子在对偶函数分段线性的分段交界处，存在多个梯度方向和松弛问题有多个最优解

   - **次梯度**：用可能的梯度方向进行凸组合表达出的方向

     $\{\Delta\mathbf{v} = \sum_{\widetilde{x}\in{T_v}}\alpha\widetilde{x}(r-\mathbf{R\widetilde{x}})$ 对任意的$\alpha\widetilde{x}\ge0$且$\sum\alpha\widetilde{x}=1\}\\T_v\triangleq\{\widetilde{\mathbf{x}}是问题(P_v)的最优解\}$ 

   - **拉格朗日对偶问题次梯度搜索算法**

     0. 初始化

        - 任选一个拉格朗日对偶问题的解$\mathbf{v}^{(0)}\ge0, l\leftarrow0$

        - 初始化原问题最优值,$\hat{v}_p\leftarrow+\infin$
        - 初始化拉格朗日对偶问题最优值， $\hat{v}_d\leftarrow-\infin$ 

     1. 拉格朗日松弛

         给定dual的当前解$\mathbf{v}^{(l)}$，求松弛问题$(P_{v^{(l)}})$ ，得到松弛问题最优解$\mathbf{x}^{(l)}$

     2. 更新当前最优解

        - $v{(P_{v^{(l)}})}\gt\hat{v}_D$, 则更新对偶问题最优值 $\hat{v}_D\leftarrow v{(P_{v^{(l)}})}, \hat{\mathbf{v}}\leftarrow\mathbf{v}^{(l)}$
        - 若$(r-\mathbf{Rx^{(l)}})\le0且\hat{v_p\lt c\mathbf{x}^{(l)}}$ ，更新原问题最优值 $\hat{v}_P\leftarrow\mathbf{cx}^{(l)}$  

     3. 算法停止条件判断

        - $(r-\mathbf{Rx^{(l)}})\le0且\mathbf{v}^{(l)}(r-\mathbf{Rx^{(l)}})=0$,此时$\mathbf{x}^{(l)}$为愿为他最优解，且$v(D_L)=v(P)$ 
        - 若进一步计算不能改进当前最优解，则停止，此时的各个解均为近似最优解

     4. 次梯度方向上的步长选择

        由一个求和发散但个体收敛的级数中选择一个步长$\lambda$ ，如$\frac{1}{2(l+1)}$ 

        $\Delta{v}\leftarrow{(r-\mathbf{Rx^{(l)}})}/||r-\mathbf{Rx^{(l)}}||$ 

        $\mathbf{v^{(l+1)}\leftarrow{v^{(l)}+\lambda\Delta{v}}}$ 

     5. 投影保证乘子的可行性

        将对偶问题的解投影至($v\ge0$) 【为了避免产生<0的乘子】

        $v_i^{(l+1)}\leftarrow\max\{0, v_i^{(l+1)}\},\forall{i} $, 跳转第1步

   [拉格朗日松弛-PythonCallGurobi示例](/Applications/CPLEX_Studio_Community129/python/examples/mp/workflow/lagrangian_relaxation.py)

###### **Dantzig-Wolfe分解**

思想：

将大量复杂的约束与多个具有易处理的特殊结构的线性约束分解开。分解得到的子问题一般具备**分块对角结构**，分块之间相互独立，并只包含一部分决策变量

利用子问题具有易处理的结构特点，将一个整体问题分解为一个限制主问题和一系列子问题

- 限制主问题是一个受限制的原问题的近似问题，通过列生成不断扩展自身以提高对原问题的近似度
- 子问题提供必要信息，协同主问题提高对原问题近似度，收敛到原问题的最优解

1. 根据极点和极方向重新建模

   - 聚焦由连接约束构成的限制主问题

   - 利用子问题可行域为凸集的特点，重新对主问题建模

     方法：将主问题的每个原始决策变量表示成各个子问题的极点($extreme\ point$)和极方向($extreme\ ray$)加权求和的形式
     $$
     具有行分割形式，主问题如下：\\
     max\quad \sum_s \mathbf{c}^{(s)}\mathbf{x}^{(s)}\\
     s.t.\quad \sum_s \mathbf{A}_s\mathbf{x}^{(s)}\le b\\
     \mathbf{x}^{(s)}\ge0\\
     利用子问题集合\mathbf{s}对应的所有可行域边界的极点和极方向，重新将它表示为如下形式：\\
     max\quad \sum_s\mathbf{c}^{(s)}(\sum_{j\in{P_s}}\lambda_{s,j}\mathbf{x}^{(s,j)}+\sum_{k\in{D_s}}{\mathbf{\mu}}_{s,k}\Delta{x}^{(s,k)})\\
     s.t.\quad \sum_s\mathbf{A}_s(\sum_{j\in{P_s}}\lambda_{s,j}\mathbf{x}^{(s,j)}+\sum_{k\in{D_s}}{\mathbf{\mu}}_{s,k}\Delta{x}^{(s,k)})\le{b}\\
     \sum_{j\in{P_s}}{\lambda}_{s,j}=1, \forall s\\
     \lambda_{s,j}\ge 0,\quad\forall{s,j\in{P_s}},\quad\mu_{s,k}\ge0,\quad\forall{s,k\in{D_s}}\\
     集合 P_s标记极点\mathbf{x}^{(s,j)},\ 集合D_s用于标记子问题极方向\Delta\mathbf{x}^{(s,k)}
     $$

     > 极方向：一个向量$d(\neq0)\in{R^n}$,满足
     >
     > $\{x\in R^n|x = x^0+\theta d,\theta\ge0\}\subset P,$对于所有$x^0\in P, P$是可行域
     >
     > 通俗点来说，解空间无界时的前进方向
     >
     > 
     >
     > 任意一个点属于多面体集F的充要条件是：$\bar{\mathbf{x}}=\sum_{j\in{P}}\lambda_jx^{(j)}+\sum_{k\in{D}}\mu_k\Delta\mathbf{x}^{(k)}, \sum_j\lambda_j=1$ 
     >
     > 即F中任意一个点都可以用其顶点和极方向线性表示出来，所有系数非负，且顶点系数之和为1

2. 子问题极点和极方向的生成方法

   在子问题中只生成可能改进限制主问题目标函数值的极点和极方向

   第 $l$ 次迭代时求解的每个子问题对应的线性规划问题：
   $$
   max\quad \bar{\mathbf{c}}^{(s)}\mathbf{x}^{(s)}-\mathbf{q}^{(l)}\\
   s.t.\quad \mathbf{T}_s\mathbf{x}^{(s)}\le\mathbf{t}^{(s)}\\
   \mathbf{x}^{(s)}\ge0\\
   其中\bar{\mathbf{c}}^{(s)} = (\mathbf{c}^{(s)}-\mathbf{v}^{(l)}\mathbf{A}_s),\\ \mathbf{v}^{(l)}为限制主问题中连接约束对应的最优对偶变量\\\mathbf{q}_s^{(l)}指当前主问题对应子问题s的凸约束最优对偶变量
   $$

**Dantzig-Wolfe分解算法**

0. 初始化

   - 迭代次数 $l\leftarrow1$
   - 初始化子问题s的极点集合 $\mathbf{P}_{s,l}\subseteq\mathbf{P}_s$
   - 初始化极方向集合 $D_{s,l}\leftarrow\emptyset$ 

1. 限制主问题求解

   求解第l次迭代得到的限制主问题，得到最优解$\lambda^{(s,l)},\mu^{(s,l)}$, 和最优对偶变量$\mathbf{v}^{(l)},\mathbf{q}^{(l)}$

2. 列生成求解

   依次对每个子问题s构造列生成，并求解子问题

   $max: v\gt0$，则在下一次迭代时将该极点$\mathbf{\bar{x}}^{(s)}$ 作为新列添加到限制主问题

   ​			$v$无界，则将对应的极方向$\Delta\mathbf{x}^{(s)}$作为新列添加到限制主问题

   否则，不存在

3. 终止条件判断

   当不存在可改进的新列时，算法停止。当前限制主问题对应的原始问题最优解和对偶问题最优解即为问题最优解：$\mathbf{x}^{(s)}\leftarrow\sum_{j\in{P_{s,l}}}\lambda_j^{(s,l)}\mathbf{x^{(j)}}+\sum_{k\in D_{s,l}}\mu_k^{(s,l)}\Delta{\mathbf{x}}^{(k)}$ 

   否则新的极点和极方向分别加入到极点集合 $P_{s,l+1}$和极方向集合 $D_{s,l+1}$, $l \leftarrow l+1$, 跳转第1步

[DW分解实现(PythonCallGurobi)](/Users/cxw/Learn/2_SIGS/ALi/Coding/OR/ScaleOptimization/DantzigWolfe.py)

###### **Benders分解**

思想：

对于一些复杂的列，通过固定部分决策变量。限制主问题根据子问题返回的决策变量值决定是否达到最优值，子问题从被固定决策变量的集合中选择可以改进限制主问题的决策变量，传递给限制主问题。

用于混合整数规划，**固定复杂的整数决策变量，保留连续的决策变量**。

1. 原模型

$$
min\quad \mathbf{cx+fy}\\
\mathbf{(BP)}\quad s.t.\quad \mathbf{Ax+Fy\ge{b}}\\
\mathbf{x\ge0,y\ge0}且为整数
$$

​	当y固定时，该问题将转变为LP问题

2. 分解策略
   $$
   min\quad \mathbf{cx+fy^{(l)}}\\
   (\mathbf{BP}_l)\quad s.t.\quad \mathbf{Ax\ge{b-Fy^{(l)}}}\\
   \mathbf{x\ge0}
   $$
   其对偶$\mathbf{BP_{y^{(l)}}}$ 

   对偶子问题
   $$
   max\quad \mathbf{v(b-Fy^{(l)})+fy^{(l)}}\\
   (\mathbf{BD}_l)\quad s.t.\quad \mathbf{vA\le{c}}\\
   \mathbf{v\ge0}
   $$
   求解对偶子问题获得两类结果：

   - 子问题最优解，对应一个极点
   - 子问题无界，对应一个极方向

   求解固定变量y的整数规划模型：
   $$
   \mathbf{min} \quad z\\
   (\mathbf{BM}_l)\quad s.t.\quad z\ge\mathbf{fy+v}^{(i)}(\mathbf{b-Fy}),\forall{i\in{P_l}}\\
   0\ge\Delta{v^{(j)}} \mathbf{b-Fy}\\
   \mathbf{y}\ge0,且为整数
   \\P_l为第一次到第l次迭代生成的极点下标集合，D_l是极方向下标集合
   $$

3. 最优性原理

   子问题$\ l\ $ 的极点为$v^{(l)}$ ，对应目标函数值为$v(BD_l)$ ，若$v(BD_l)$ 比上次迭代得到的限制主问题最优目标函数值小，即 $v(BD_l)\le{v(BM_{l-1})}$,则$v^{(l)}$为主问题(BM)和原始问题BP的最优解

**Benders分解算法**

0. 初始化

   - 初始化极点集合 $P_0\leftarrow \emptyset$; 极方向集合$D_0\leftarrow\emptyset$
   - 主问题目标函数值: $z_0 = -\infin$ 
   - 满足BP的初始解$y^{(0)}$
   - 迭代次数 $l\leftarrow1$

1. Benders子问题求解

   求解Benders子问题$(BD_{y^{(l-1)}})$ 

   - 问题有界，则获得极点，跳转第2步
   - 问题无界，则获得极方向，跳转第3步

2. 算法停止判断

   - 若子问题的解满足$\mathbf{fy}^{(l)}+\mathbf{v}^{(l)}(\mathbf{b-Fy}^{(l)})\le{v(BM_{l-1})}$ ，算法停止，$y^{(l)}$为$(BP)$最优解

   - 求解$BD_{y^{(l)}}$，可获得决策变量$\mathbf{x}^{(l)}$ 

3. 限制主问题更新

   - 若第2步得到新的极点，则将该极点加入值极点集合中
     - 并在限制主问题($BM_l$)中增加约束条件：$z\ge\mathbf{fy+v}^{(l)}\mathbf{b-Fy}$ 
   - 若第2步得到新的极方向，则将该极方向加入极方向集合中
     - 并在限制主问题($BM_l$)中增加约束条件：$0\ge\Delta{v^{(l)}} (\mathbf{b-Fy})$ 

4. 主问题求解

   - 精确或近似求解限制主问题($BM_l$)得到解$y^{(l)}$和目标函数值$z_l$
   - $l \leftarrow{l+1}$, 返回第1步

[Benders分解示代码实现(PythonCallGurobi)](/Users/cxw/Learn/2_SIGS/ALi/Coding/OR/ScaleOptimization/Benders.py)



##### chp14 计算复杂性理论

算法复杂度：该算法完成一个优化问题计算素偶需要的时间上界，是关于实例规模的函数 $O(\cdot)$ 

多项式时间：当一个实例问题可以用多项式时间算法求解，则，该优化问题是**完全可解**的

判定问题：1.受限的可行性问题是否存在一个解满足该实例的所有约束条件，该问题为一个判定问题

​				   2.给一个阈值$v$, 受限的阈值问题为是否存在一个解满足约束条件，并且目标值不会比$v$差 

$\mathbf{P}$问题 $\triangleq$ {多项式时间可解的判定问题}

$\mathbf{NP}$问题$\triangleq$ {非确定多项式时间可解的判定问题}，  $\mathbf{P}\subset\mathbf{NP}, \mathbf{P}\subset${多项式时间可解问题}，

判定问题包括 判定问题和不可判定问题，判定问题包括：多项式时间可解判定问题P，非确定多项式时间可解判定问题NP

多项式时间归约：一类问题属于另一类(网络流问题属于线性规划)

$\mathbf{NP}$完全问题：NP问题中可以进一步归约至该子类问题，称盖子类为NP-complete问题

​	$\mathbf{NP}-complete\triangleq\{\mathbf{(Q)}\in\mathbf{NP}:NP中的每个问题都可以归约至Q\}$

$\mathbf{NP}$难问题：

​	$\mathbf{NP}-hard\triangleq\{(\mathbf{Q}):NP完全类中的一些问题可以归约至\mathbf{Q})\}$



##### chp14 离散优化的启发式算法

1. 构造型启发式(Constructive search)

   - 贪婪选择

     每次迭代固定下一个变量，并在临时解变量已固定的情况下，可以保证下一个解的最大可行性，并最大限度改进目标函数值

2. 针对离散优化INLPs问题改进搜索启发式算法

   - 邻域搜索
   - 多起点搜索

3. 元启发式

   允许非改进的可行移动来跳出局部最优，[但会产生搜索的无限循环]

   - 禁忌搜索

     0. 初始化 选择初始可行解和迭代次数
     1. 停止 当前解的移动集合中没有**非禁忌移动**能得到更好解，则停止迭代
     2. 移动 选择非禁忌移动$\Delta{x}$
     3. 逐步操作  $x^{(t+1)}\leftarrow x^{(t)}+\Delta{x}^{(t+1)}$
     4. 现有解 若当前解更优，则替换
     5. 禁忌列表从禁忌列表中移除足够次数的移动，并添加从解$x^{(t+1)}$立即返回解$x^{(t)}$的所有集合
     6. 增量 $t\leftarrow t+1$ 

   - 模拟退火

     依概率来接受非改进移动的方式

4. 进化元启发式

   - 遗传算法



##### chp15 无约束的非线性规划

1. 无约束非线性规划模型

   连续可导时

2. 一维搜索

   - 单峰目标函数

     每个无约束局部最优就是全局最优

   - 黄金分割搜索

     迅速缩小包含最优点的区间

   - 二次拟合搜索

3. 局部最优条件

   $x^{(t+1)}\leftarrow x^{(t)}+\lambda \Delta{x}$

   - 一阶导数和梯度

     描述了1维或n维变量函数f在当前决策变量处作微小变动的变化速率

   - 二阶导数和海塞矩阵

     描述了f在当前决策变量初的邻域的曲率或斜率的变化

   - 单变量泰勒近似

     $f(x^{(t)}+\lambda) = f(x^{(t)})+\frac{\lambda}{1!}f'(x^{(t)})+\frac{\lambda^2}{2!}f''(x^{(t)})+\frac{\lambda^3}{3!}f'''(x^{(t)})+\dots$

     $f_1(x^{(t)}+\lambda) \triangleq f(x^{(t)})+{\lambda}f'(x^{(t)})$

     $f_2(x^{(t)}+\lambda) \triangleq f(x^{(t)})+{\lambda}f'(x^{(t)})+\frac{\lambda^2}{2}f''(x^{(t)})$

   - 多维变量的泰勒近似

     $f_1(\mathbf{x}^{(t)}+\lambda\Delta\mathbf{x})\triangleq f(\mathbf{x}^{(t)})+\lambda \nabla f(\mathbf{x}^{(t)})\cdot \Delta\mathbf{x} \triangleq f(\mathbf{x}^{(t)})+\lambda \sum_{j=1}^n(\frac{\partial f}{\partial x_j})\Delta x_j$

     ...

   - 驻点和局部最优

     - 驻点：所有一阶导等于0的点
     - 鞍点：局部最大点和局部最小点之外的驻点
     - 海塞矩阵：在局部最大点是半负定的，在局部最小点是半正定的

4. 凹凸函数和全局最优

   - 凹函数和凸函数

   - 凹凸函数的判定

     海塞矩阵为半正定时，为凸函数；为半负定时，为凹函数

     - 线性函数既是凹函数也是凸函数

     - 凸函数的非负加权平方和还是凸函数；凹函数的非负加权平方和还是凹函数
     - 凸函数的$max$还是凸函数，凹函数的$min$还是凹函数
     - $g(h(x))$ g和h均为凸时，该函数也为凸

5. 梯度搜索

   - 梯度搜索算法(最速下降)

     $\Delta\mathbf{X}\triangleq \pm\nabla f(\mathbf{x}^{(t)}), max(+),min(-)$

     算法步骤：

     0. 初始化 
     1. 梯度
     2. 驻点 $||\nabla f(x^{(t)})||\lt \varepsilon $，停止 
     3. 方向
     4. 线性搜索  求解$max/min\ f(x^{(t)}+\lambda\Delta{x}^{(t+1)})$ 获得步长
     5. 新点
     6. 前进

     缺点：在最优解附近，很小步长会带来巨大变化，收敛性差，曲折性

6. 牛顿法

   - 牛顿步

     对$\Delta x$求偏导，步长定为1

     $\nabla^2 f(\mathbf{x}^{(t)})\Delta{\mathbf{x}}=-\nabla{f(\mathbf{x}^{(t)})} \\\Delta{\mathbf{x}}=-[\nabla^2f(\mathbf{x}^{(t)})]^{-1}\nabla{f(\mathbf{x}^{(t)})}$

   - 牛顿法算法步骤

     0. 初始化 
     1. 导数 计算一阶导和二阶海塞矩阵
     2. 驻点
     3. 牛顿步
     4. 新点
     5. 前进

   - 算法特点
     - 优势：不需要线性搜索
     - 劣势：二阶泰勒近似的计算量较大；有可能无法收敛；要求海塞矩阵的方程组非奇异

7. 拟牛顿法和BFGS搜索

   1. 偏转矩阵

      $\Delta{\mathbf{x}}=-[\nabla^2f(\mathbf{x}^{(t)})]^{-1}\nabla{f(\mathbf{x}^{(t)})}$

      牛顿法是对当前梯度应用合适的偏转矩阵$[\nabla^2f(\mathbf{x}^{(t)})]^{-1}$ 来计算方向

   2. 拟牛顿方式

      采用的偏转矩阵: $D_{t+1}g=d, d=x^{(t+1)}-x^{(t)},g=\nabla{f(x^{(t+1)})-\nabla{f(x^{(t)})}}$ 

      $D_{t+1}\leftarrow D_t + (1+\frac{gD_tg}{d\cdot{g}})\frac{dd^T}{d\cdot{g}}-\frac{D_tgd^T+dg^TD_t}{d\cdot{g}}$ 



##### chp17 带约束的非线性规划

1. 特殊的NLP

   1. 凸规划

      目标函数为最大化凹函数f或最小化凸函数f, 每个满足$\le$约束都是凸函数, 每个满足$\ge$约束的都是凹函数，满足=约束的g 都是线性的。

   2. 可分离规划

   3. 二次规划

      目标函数是二次型

2. **拉格朗日乘子法**

   **要求求解的NLP的约束为等式**

   ==需要验证驻点的x部分是否是最优的==

   局限性：驻点条件仅在线性或简单非线性才容易求解

   ​				对于不等式约束问题，难以确定哪些约束是紧约束

   ​				

3. **KKT条件**

   拉格朗日乘子可以反映右端项$b_i$变化时最优目标函数值变化率

   KKT点：某个x，存在一个对应的 v 联合起来满足KKT条件：

   - 互补松弛条件

     $v_i[b_i-g_i(\mathbf{x})]=0$ , 对于所有不等式约束

   - 约束条件

     原$NLP$问题的约束

   - 符号约束

   | 目标函数 | $\le$     |   $\ge$   |
   | -------- | --------- | :-------: |
   | $min$    | $v_i\le0$ | $v_i\ge0$ |
   | $max$    | $v_i\ge0$ | $v_i\le0$ |

   - 梯度方程：

     $\sum_i\nabla{g_i(\mathbf{x}})v_i=\nabla{f(\mathbf{x})}$ 

   > 条件成立：
   >
   > 某个点如果没有可行改进方向，则该点可以视为局部最优
   >
   > 一阶泰勒级数： $f(x+\Delta{x})=f(x)+f'(x)\Delta{x}$ ,是否改进取决于$\nabla{f(x)}\cdot{\Delta{x}}$的符号
   >
   > 当$\nabla{f(x)}\cdot{\Delta{x}}=0$时，需要更多信息来确定
   >
   > 所以为了保持约束可行，$\nabla{g_i(x)}\cdot{\Delta{x}}$要与原约束同号

   KKT条件是一种一阶的检验可行改进方向是否存在的方法

4. 惩罚与障碍法

   1. 惩罚函数法

      $max(min)\ F(x)\triangleq f(x)\pm \mu\sum_ip_i(x), max:+,min:-$

      $\mu为惩罚乘子, p_i(x)=0,当x满足约束i, 否则>0$  

      常用惩罚函数：

      - $max\{0,\ b_i-g_i(x)\},\ max^2\{0,\ b_i-g_i(x)\}, 对于\ge约束$
      - $max\{0,\ g_i(x)-b_i\},\ max^2\{0,\ g_i(x)-b_i\}, 对于\le约束$
      - $|g_i(x)-b_i|,\ |g_i(x)-b_i|^2, 对于=约束$ 

      平方的惩罚函数是可微的

      - 序列无约束惩罚技术

        惩罚因子应先取一个较低的值，然后随着计算过程的继续，逐渐增加

        $\beta$为一个增大因子

        0. 初始化

        1. 无约束最优化

        2. 停止

        3. 增加

           $\mu_{t+1} \leftarrow \beta\cdot\mu_t$

   2. 障碍法

      保证选择的初始可行点的移动不离开可行域

      $max(min)\ F(x)\triangleq f(x)\pm \mu\sum_ip_i(x), max:+,min:-$

      $\mu$为正的障碍乘子，$q_i$满足：$q_i(x)\rightarrow\infin$ ,当约束i趋向于起作用

      常用的障碍函数：

      - $-\ln[g_i(x)-b_i],\ \frac{1}{g_i(x)-b_i},\ 对于\ge约束$
      - $-\ln[b_i-g_i(x)],\ \frac{1}{b_i-g_i(x)},\ 对于\le约束$,    在函数接近边界时，障碍函数将趋向$+\infin$ 

      **当最优解出现在边界上时，障碍函数法的最优解一定不是原问题最优解**

      随着$\mu\rightarrow0$，障碍函数法的最优解将收敛至原问题最优解

      - 序列无约束障碍技术

        障碍乘子应从较大的$\mu\ge0$开始，随着搜索逐渐减小

        $\beta$为一个减小因子

        0. 初始化

        1. 无约束最优化

        2. 停止

        3. 增加

           $\mu_{t+1} \leftarrow \beta\cdot\mu_t$

5. 既约梯度法(reduced gradient)

   **前提：$NLP$的约束为线性约束**

   添加松弛变量以变成标准等式形式

   变量包括：基变量，取值为0的非基变量，取值为正数的超基变量(superbasic)

   $\Delta{\mathbf{x}^{B}}=-\mathbf{B^{-1}N\Delta{x^{(N)}}}$ 

   既约梯度：$\nabla{f(\mathbf{x})\cdot\Delta{\mathbf{x}}}=(\nabla{f(\mathbf{x}^{(N)})-\nabla{f(\mathbf{x})^{(B)}\mathbf{B^{-1}N}}})\Delta{\mathbf{x}}$ 前面的系数即既约梯度

---

### 

## 2 Heuristics

### neighborhood search

- [x] 迭代搜索
- [x] [模拟退火][SA]
- [x] [禁忌搜索][TS]
- [x] [迭代局部搜索][ELS]
- [x] [变邻域搜索][VNS]
- [ ] ==[自适应大邻域搜索]()==  ★★★★★



---

[ELS]: 在局部搜索的最优解上增加扰动，重新局部搜索

---

[SA]: 移动后的解比当前解要差，也以一定的概率接受移动，但该概率随着时间推移逐渐降低。

---

[TS]: 对已经历过的搜索过程进行记录，从而指导下一步的搜索方向。邻域搜索(locals_earch)基础上设置禁忌表(tabu_list)

​	产生邻域解，由于当前最好解，不考虑是否被禁忌而直接替换当前解，并加入禁忌表；如果不优于当前解，则在所有候选解中选出不在禁忌状态下的最好解作为新解，将对应操作加入禁忌表。

---

[VNS]: 搜遍当前邻域结构无更优解，则更换邻域；若有更优解则更新当前解从第一个邻域重新开始

​	存在一个shaking，实现对当前解的一个扰动。

### 群智能

- [x] 遗传算法
- [ ] 蚁群算法
- [ ] 粒子群算法



## 3 ML

### <Python机器学习>

[Scikit-learn](http://www.scikitlearn.com.cn) 

#### 概览

- 有监督学习

  - 分类: 离散分类标签
  - 回归: 连续信号标签

- 无监督学习

  - 降维
  - 聚类

- 强化学习

  决策过程, 奖励机制, 学习一系列行动

原始数据, 标签→ 训练集, 测试集 → ML → Model → 新数据[预测]:标签

**交叉验证**

#### chp2 训练简单的机器学习分类算法

##### 1.人工神经元

###### 神经元

输入: $x,\ 权重w,\ z =w_0x_0+w_1x_1+\dots+w_mx_m $ 

单位阶跃函数: $\phi(z) = 1, z\ge\theta, else\ -1$ 

给出偏置 $w_0=-\theta,x_0=1,\\ \phi(z)=\phi(w_0x_0+w_1x_1+\dots+w_mx_m)=\phi(w^{T}x)=1,z\ge0; -1, else$

##### 2.感知器学习规则

> 1. 把权重初始化为0或很小的随机数
>
> 2. 对每个训练样本$x^{(i)}$
>
>    a. 计算输出值$\hat{y}^{(i)}$
>
>    b. 更新权重

$w_j:=w_j+\Delta{w_j}\\\Delta{w_j} = \eta(y^{(i)}-\hat{y}^{(i)})x^{(i)}, 0\le\eta\le1为学习率$ 

##### 3.自适应神经元和学习收敛

自适应线性神经元(Adaline)， 权重更新基于线性激活函数

其线性激活函数为$\phi(z)=\phi(w^{T}x)=w^{T}x$ 

1. 梯度下降为最小代价函数

   $SSE:\ J(w) = \frac{1}{2}\sum_i(y^{(i)}-\phi(z^{(i)}))^2$ 

   $\Delta{w}=-\eta\nabla{J}(w),\\\Delta{w_j}=-\eta\frac{\partial{J}}{\partial{w_j}}=\eta\sum_i(y^{(i)}-\phi(z^{(i)}))x_j^{(i)}$ 

2. 通过调整特征大小改善梯度下降

   标准化: $x_j = \frac{x_j-\mu_j}{\sigma_j}$ 

   `X_std[:, 0] = (X[:, 0].means()) / X[:, 0].std()` 

3. 大规模机器学习与随机梯度下降

   随机梯度下降法： 随机选择部份样本进行梯度计算

#### chp3 scikit-learn机器学习分类器

##### 1.训练感知器

当类不是完全可分时， 感知器将永远不收敛

分类标签编码为整数`np.unique(iris.target)`

数据集分割: 

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 1, stratify=y) # stratify = y 按y标签进行分层抽样 np.bincount(y)
# 标准化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform()
# 调用sklearn中的感知器
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
# 观差预测效果
y_pred = ppn.predict(X_test_std)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
ppn.score(X_test_std, y_test) # 或者调用分类器的评分方法
```

##### 2.逻辑回归

**[Logit: 分类模型，而不是回归模型]**

###### 逻辑回归

让步比$: \frac{p}{(1-p)}, p:阳性事件的概率$

logit函数: $logit(p) = \ln\frac{p}{(1-p)}, \\logit(p(y=1|\mathbf{ \mathit{x}}))=w^Tx$ 

逆形式: $\phi(z) = \frac{1}{1+e^{-z}}$ 

###### 逻辑代价函数的权重

使用极大似然来作为目标函数

$l(w)=\ln(L(w))=\sum_{i=1}^n[y^{(i)}\ln{\phi(z^{(i)})+(1-y^{(i)})\ln(1-\phi(z^{(i)}))}]$ 

Lasso回归就是为最小化损失函数增加一个 **模型参数$w$服从0均值拉普拉斯分布**

​	逼迫更多的$w_i$为0，变稀疏【==自动实现了特征选择==】

Ridge回归就是为最小化损失函数增加一个 **模型参数$w$服从均值正态分布**

​	惩罚权重变大的趋势

逻辑回归在0附近敏感，原理0点的位置不敏感，逻辑回归模型更加关注分类边界

###### 正则化解决过拟合

过拟合意味着有较高的方差，参数太多，模型过于复杂

##### 3.支持向量机

寻找最大化边界

###### 最大边际

间隔较小容易过拟合

###### 松弛变量处理线性可分

变量c来控制对分类错误的惩罚，较小的c意味着对分类错误的要求不严格，获得较大的分间隔宽度

减小c可以增加偏执，降低模型方差 

##### 4.核支持向量机

**核**方法的逻辑：针对现行不可分数据，建立非线性组合，通过映射函数将原始特征映射到一个高维空间，使得特征在该空间下线性可分

##### 5.决策树

决策树从树根开始，在信息增益(IG)最大的节点上分裂数据

###### 获取最大收益

目标函数能够最大化每次分裂的信息增益

$IG(D_p,\ f)=I(D_p)-\sum_{j=1}^m \frac{N_j}{N_p}I(D_j)\\f为分裂数据的特征, D_p为父节点, D_j为第j个子节点,N_p为父节点样本数,N_j为第j个子节点的样本数,I为杂质含量 $

子节点不纯度量越低，信息增益越大

三种不纯度(impurity)度量

> 不纯度值：当前节点的不纯度值，反映了当前节点的预测能力，减少错误分类概率的判断标准
>
> $I(X) = I(\frac{|x,c(x)\in{c_1}|}{|X|},\dots,\frac{x,c(x)\in{c_k}}{|X|}), 分别表示每类在样本总数中的占比$ 
>
> 满足如下：
>
> 1. 当所有样本同属于一类时，$I$取最小值
> 2. 每个类下样本总数相同时，$I$取最大值
> 3. I关于每个取值对称
> 4. I是一个凸函数
>
> $\Delta{I}(X,\{X_1,\dots,X_s\})=I(X)-\sum_{j=1}^s \frac{|X_j|}{X}\cdot I(x_j)$ 
>
> 一个好的划分应该让不纯度变低, 所以**决策树节点划分的依据：找到一个特征的取值，使该划分是不纯度缩减量最大**



- 信息墒

  信息墒：$H(A)=-\sum_{j=1}^kp(A_i)\log_2(P(A_i))$，A有k个可能的输出

  $I_H(t)=-\sum_{i=1}^cp(i|t)\log_2{p(i|t)}, p(i|t)$为节点t属于c类样本的概率

- 基尼指数

  $I_G(t)=\sum_{i=1}^cp(i|t)(1-p(i|t))=1-\sum_{i=1}^cp(i|t)^2$

- 分类误差

  $I_E=1-max\{p(i|t)\}$

###### 构建决策树

```python
from 	sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
```



##### 6.随机森林

决策树的集合—对分别受方差影响的多个决策树取平均值，以建立一个更好的泛化性能和不易过拟合的强大模型

**随机森林算法**：

		- 随机提取规模为n的引导样本(从训练集中随机选择n个可替换样本)
  - 基于引导样本的数据生成决策树。在每个节点进行
    		- 随机选择d个特征，无需替换
        		- 根据目标函数提供的最佳分裂特征来分裂节点，例如最大化信息增益
- 重复步骤1, 2 k次

```python
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=2, n_jobs=2)
forest.fit(X_train, y_train)
```

##### 7.K-NN算法

基于实例的学习，属于非参数模型。通过记忆训练集来完成任务

新数据点的分类标签式由最靠近该点的k个数据点的多数票来决定的

**KKN算法**：

	- 选择k个数和一个距离度量
	- 找到要分类样本的k-近邻
	- 以多数票机制确定分类标签

```python
from sklearn.neighbors import KNeighborsClassifier
kkn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
kkn.fit(X_train_std, y_train) # p=2,欧几里得距离; p=1,曼哈顿距离
```

#### chp4 数据预处理

##### 1. 处理缺失数据

1. 识别缺失值

   `df.isnull.sum()`获得每一列缺失数值统计

   `df.values`获得DataFrame底层的numpy数据

2. 删除缺失值

   > DataFrame.dropna(*self*, *axis=0*, *how='any'*, *thresh=None*, *subset=None*, *inplace=False*)

   `df.dropna(axis=0)` 删除行

3. 填补缺失数据

   均值插补

   ```python
   from sklearn.preprocessing import Imputer
   imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)  # strategy: most_frequent, constant
   imp.fit(data)
   imp.transform(X_test)
   ```

##### 2. 处理分类数据

###### 1.名词特征和序数特征

###### 2.映射数据特征

人工定义映射关系

```python
size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
```

###### 3.分类标签编码

```python
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['class_label']))}
# call sklearn
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['class_label'].values)
```

###### 4.为名词特征作热编码

避免整数化编码带来数值大小的影响

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0]) # 该参数为想要变换的列的位置
ohe.fit_transform(X).toarray()
# 使用pandas的get_dummies方法
pd.get_dummies(df[['first', 'second']], drop_first=True) # 通过删除第一列，避免多重共线性
```

##### 3.分割数据集

##### 4.特征保持在同一尺度上

###### 归一化

最大最小比例调整：$x_{norm}^{(i)}=\frac{x^{(i)-x_{min}}}{x_{max}-x_{min}}$

```python
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fir_transform(X_train)
```

###### 标准化

$x_{std}^{(i)} = \frac{x^{(i)}-\mu_x}{\sigma_x}$

```python
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transfrom(X_train)
```

##### 5.选择有意义的特征

在训练集上表现较好，很可能发生了过拟合，减少泛化误差的方法：

收集更多数据; 通过正则化引入对复杂性的惩罚; 选择参数较少的简单模型; 减少数据维数

L1正则化, L2正则化

###### 降维

特征选择, 特征提取

经典特征选择算法—逆顺序选择(SBS)

​	对分类器性能营销最小的衰减来降低初始特征子空间的维数/每步去除因特征去除损失最小的那个特征

##### 6.随机森林来评估特征的重要性

```python
forest.feature_importances_
```

而后使用SelectFromModel来选择特征

```python
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
```

#### chp5 降维

特征选择保持原始特征，而特征提取将数据投影至新的特征空间

##### 1.主成分分析[无监督需学习]

本质：寻找高维数据中存在方差最大的方向

$x \leftarrow xW$

**PCA算法步骤**：

1. 标准化d维数据集

2. 构建协方差矩阵

   协方差: $\sigma_{jk}=\frac{1}{n}\sum_{i=1}^n(x^{(i)}_j-\mu_j)(x^{(i)}_k-\mu_k)$

   `cov_mat = np.cov(X_train_std.T)`

3. 将协方差矩阵分解为特征向量和特征值

   `eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)`

4. 通过降阶对特征值排序，对相应的特征向量排序

    特征值方差解释比: $\frac{\lambda_j}{\sum_{j=1}^d\lambda_j}$

   `eigen_pairs = [np.abs(eigen_vals[i], eigen_vecs[:, i]) for i in range(len(eigen_vals))] `

   `eigen_pairs.sort(key=lambdak:k[0], reverse=True)`

5. 选择对应k个最大特征值的k个特征向量，其中k维新子空间的维数($k\le{d}$)

6.  从最上面的k特征向量开始构造投影矩阵W

   `W = np.hstack((eigen_pairs[0][1][:, np.newaxis],eigen_pairs[1][1][:, np.newaxis]))`

7. 用投影矩阵W变化d维输入数据集X，获得新的k维特征子空间

   `X_train_pac = X_train_Std.dot(W)`

```python
from sklearn.deomposition import PCA
pca = PCA(n_components=2)
# 获得解释方差比 pca.explained_variance_ratio_
X_train_pca = pca.fit_transform(X_train_std)
```

##### 2.线性判别分析[有监督学习]

本质：寻找和优化具有可分性的特征子空间

前提：LDA假设数据呈正态分布

**LDA算法步骤**：

1. 标准化d维数据集
2. 计算每个类的d维钧之向量
3. 构建跨类的散布矩阵$S_B$和类内部的散布矩阵$S_W$
4. 矩阵$S_w^{-1}S_B$计算特征向量和对应的特征值
5. 按特征值降序排列，并对特征向量排序
6. 选择对应于k个最大特征值的特征向量，构建d$\times$k维变换矩阵W
7. 把变化矩阵W投射到新的特征子空间

###### 散布矩阵

每个均值向量存储着对应分类样本i的特征平均值

$S_i=\sum_{x\in{D_i}}^c(x-m_i)(x-m_i)^T,\ S_W = \sum_{i=1}^cS_i$ 

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
```

##### 3.非线性映射的核主成分分析

本质：核PCA非线性映射至高维空间，然后用标准PCA将数据投影回低维空间

定义非线性映射函数$\phi:\ \mathbb{R}^d\rightarrow \mathbb{R}^k(k\ggg{d})$

核：

 -  径向基函数(RBF, 高斯核函数)

    $\gamma=\frac{1}{2\sigma}$

	- 多项式核

```python
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)
```

#### chp6 模型评估和超参数调优

##### 1.Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))
pipe_lr.fit(X_train, y_train)
```

##### 2.交叉验证

###### 抵抗交叉验证

- 数据集
  - 训练集
    - 训练集
    - 验证集：用于模型评估
  - 测试集

###### k折交叉验证

将训练集随机分裂为k个无更换子集，k-1个用于模型训练，一个用于模型评估。重复该过程n个，得到k个模型和k次性能估计

**k的最优标准值为10**

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jops=1)
```

##### 3.用学习曲线和验证曲线 调试算法

###### 学习曲线诊断偏差和方差问题

遇到的问题：

 -  高偏差

    训练和交叉验证准确度均低，**[欠拟合]**

    方法：增加模型参数个数

 -  高方差

    模型在训练和交叉验证的准确度差异大，**[过拟合]**

    方法：减小模型复杂度，增大正则化参数，特征选择或特征提取

```python
from model_selection import learning_curve
train_size, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_size=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)
```

###### 验证曲线来解决欠拟合和过拟合问题

```python
from model_selection import validation_curve
param_range = [0.001, 0.01, 0.1, 1, 10, 100]
train_size, train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y=y_train, param_name='logisticregression_C', param_range=param_range,cv=10,n_jobs=1)
```

##### 4.通过网格搜索来调优

通过寻找最优超参数组合来进一步提高模型性能

两类参数：

- 训练数据中学习到的参数，如逻辑回归的权重
- 单独优化的算法参数：模型的调优参数/超参数， 如决策树的深度

```python
from sklearn.model_selection import GridSearchCV
param_range = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = [{'svc_C':param_range, 'svc_kernel'=['linear']},
               {'svc_C':param_range, 'svc_gamma':param_range, 'svc_kernel':['rbf']}]
gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
gs = gs.fit(X_train, y_train)
gs.best_scores_
gs.best_params_
clf = gs.best_estimator_
```

##### 5.比较不同的性能评估指标

###### 含混矩阵分析

```python
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
```

###### 准确度和召回率

误差率(ERR), 准确率(ACC)

精度(PRE), 召回率(REC) = 真阳率

```python
from sklearn.metrics import precision_score, recall_score, f1_score
recall_score(y_true=y_test, y_pred=y_pred)
```



#### chp7 综合不同模型的组合学习

##### 1.集成学习

多数票机制

套袋机制

自适应增强`from sklearn.ensemble import AdaBoostClassifier`



#### chp10 回归分析预测连续目标变量

##### 1.线性回归

最小二乘：$w = (X^TX)^{-1}X^Ty$ 

```python
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)

```

##### 2.RANSAC拟合稳健回归模型

随机抽样一致性(RANSAC)算法步骤：

1. 随机选择一定数量的样本作为内点来拟合模型
2. 用模型测试所有其他的点， 把落在用户给定容限范围内的点放入内点集
3. 调整模型中使用的所有的内点
4. 用内点重新拟合模型
5. 评估模型预测结果与内点集相比较的误差
6. 性能达到目标或到最大迭代次数，则停止迭代

```python
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, loss='absolute_loss', residual_threshold=5.0,random_state=0)
ransac.fit(X, y)
```

##### 3.回归模型性能评估

残差图

均方误差(MSE)

拟合优度($R^2$) $R^2=1-\frac{SSE:平均误差之和}{SST:平方和总和}$ 

##### 4.正则化方法回归

岭回归 L2范数惩罚

LASSO回归 L1范数惩罚

##### 5.多项式回归

```python
from sklearn.preprocessing import PolynomialFeatures
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)
```

##### 6.随机森林处理非线性关系

###### 决策树回归



#### chp11 聚类

##### 1.k-means

算法步骤：

1. 随机从样本中挑选k个重心座位初始聚类中心
2. 将每个样本分配到最近的重心$\mu^{(j)}, j\in\{1,\dots,k\}$
3. 将重心移到已分配样本的中心
4. 重复步骤2, 3，直到集群赋值不再改变或到达最大迭代数

**可以使用误差平方和(SSE)来评价聚类性能**  `km.inertial_`

###### k-means ++

根据距离来随机初始化重心

```python
from sklearn.cluster import KMeans
km = KMeans(n_cluster=3, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=0)
```

###### 硬聚类和软聚类

模糊C-均值聚类

###### 聚类数确定

k增大时，SSE失真会减小，硬选择失真增速最快时的k值

###### 用轮廓图量化聚类质量

轮廓系数，理想为1， 即其他类与该样本所在类分离度较大

计算单个样本的轮廓系数：

1. 计算集群内的内聚度$a^{(i)}$, 即样本$x^{(i)}$与集群内所有其他点的平均距离

2. 计算集群与最近集群的分离度$b^{(i)}$, 即样本$x^{(i)}$与最近集群内所有样本的平均距离

3. 计算轮廓系数$s^{(i)}$, 即集群内聚度和集群分离度的差

   $s^{(i)}=\frac{b^{(i)}-a^{(i)}}{max\{b^{(i)},a^{(i)}\}}$ 

```python
from sklearn.metrics import silhouette_samples
y_km = km.fit_predict(X)
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
```

##### 2.层次聚类

###### 凝聚层次聚类

单连接：计算两个集群中最相似成员之间的距离，然后合并两个集群

全连接：计算两个集群中最不相似成员之间的距离，然后合并两个集群

平均连接法：基于两个集群中所有组成员之间的最小平均距离来合并集群

沃德连接法：合并引起群内总SSE增长最小的两个集群

```python
from sklearn.cluster import agglomerativeCLustering
ac = agglomerativeCLustering(n_cluster=3,, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(X)
```

##### 3.DBSCAN定位高密度区域

密度：在指定半径$\varepsilon$范围内的点数

核心点：有指定数量(MinPts)的相邻点落在以该点为圆心的指定半径$\varepsilon$范围内，则该点为核心点

边界点：在核心点半径$\varepsilon$范围内，相邻点比半径$\varepsilon$范围内的MinPts少

噪声点

算法：

1. 每个核心点或连接的核心点组成单独的集群
2. 把每个边界点分配到与其核心点相对应的集群

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)
```



#### chp12 多层神经网络

##### 1.用人工神经网络为复杂函数建模

###### 单层神经网络

基于训练集的梯度，梯度相反方向 来更新模型权重，来优化SSE的目标函数$J(w)$找到最优权重

激发函数：$\phi(z)=z=a,\ z=w^Tx$

###### 多层神经网络

多层感知器(MLP): 一个输入层、一个隐藏层、一个输出层

$a^{(l)}_i:$ 第l层的第i个活化单元

$l$层中的每个单元通过权重系数与$l+1$层的所有单元连接

如$w^{(l)}_{k,j}:l$ 层的第k个单元与l+1层的第j个单元的连接

输入层到隐藏层的权重矩阵：$\mathbf{W}^{(h)}$,隐藏层到输出层的权重矩阵：$\mathbf{W}^{(out)}$

###### 正向传播激活神经网络

步骤：

1. 从输出层开始，把训练数据的恶模型通过网络传播出去
2. 基于网络输出，用稍后将描述的代价函数计算想要的最小化错误
3. 反向传播误差，匹配网络中相应权重并更新模型

$z^{(h)}_1= a^{(in)}_0w^{(h)}_{0,1}+a^{(in)}_1w^{(h)}_{1,1}+\dots+a^{(in)}_mw^{(h)}_{m,1}$

$a^{(h)}_1=\phi(z_1^{(h)})$ 

可以将神经元看作逻辑回归单元(S单元)

##### 2.训练人工神经网络

###### 逻辑成本函数的计算

$J(w)=-\sum_{i=1}^ny^{[i]}\log(a^{[i]})+(1-y^{[i]})\log(1-a^{[i]}), a^{[i]}=\phi(z^{[i]})$

> 链式求导
>
> $F(x)=f(g(x))=z,\ x\rightarrow g\rightarrow f\rightarrow z$

计算$\delta^{(h)}$层的误差矩阵：$\delta^{(h)}=\delta^{(out)}(W^{(out)})^T\odot(a^{(h)}\odot(1-a^{(h)}))$

反向传播更新权重：$W^{(l)}:=W^{(l)}-\eta{\Delta^{(l)}}$ 



#### chp15 深度卷积神经网路(CNN)

##### 1.构建卷积神经网络的模块

###### CNN

通过逐层组合底肌特征来构造**特征层次**

稀疏连通， 参数共享：输入图像的不同区域使用相同的权重

CNN通常由若干卷积层(conv)和子采样层(也称为池层(P))所组成，尾随着一个或多个全连接层(FC).全连接层本质上是多层感知器

###### 离散卷积

一维离散卷积：$y=x_{(输入)}*w_{(过滤器/核)}\rightarrow y[i]=\sum_{k=-\infin}^{+\infin} x[i-k]w[k]$

0的填充模式：

- 完全模式：填充参数p=m-1，增加了向量维数
- 相同模式：希望输出与输入向量大小相同
- 有效模式

```python
np.convolve(x, w, mode='same')
```

矩阵转换与转制不同：`W_rot = W[::-1,::-1]`

子采样：最大池，平均值



#### chp16 递归神经网络(RNN)

为序列数据建模

##### 1.序列数据

##### 2.RNN

递归神经网络的隐藏层由输入层和先前迭代的隐藏层获取其输入

隐藏层中相邻迭代的信息流允许网络对事件保持记忆

单层RNN中的不同权重矩阵：$W_{xh},W_{hh},W_{hy}$ 输入层到隐藏层，递归遍相关联，隐藏层到输出层

## 4 CS

### CSAPP

#### ch1 计算机系统漫游

---

> **编码**
>
> - **Ascii** 定义 128个字符[英文、数字、符号等]
>
> - **ANSI** 扩展的Ascii编码，表示英文时用1个字节，表示中文2个字节
> - **Unicode** 汉字表示习惯双字节，都是用两个字节来表示的编码方式；对英文字符浪费了1个字节
> - **UTF-8** 针对Unicode的可变长度字符编码

> ---
>
> **编译系统**
>
> 	- 预处理器: 预处理将#命令，修改原始程序 → hello.i 
> 	- 编译器: 将hello.i翻译层呢文本文件hello.s
> 	- 汇编器: 汇编器将hello.s翻译成机器语言指令，指令打包成“可重定位目标程序”的格式 hello.o
> 	- 链接器: 链接标准库内的函数，并合并到hello.o中，形成可执行目标文件

> ---
>
> **系统硬件组成**
>
> - 总线 8字节=64位
> - I/O
> - 主存
> - CPU  含有一个大小为1个字的存储设备(寄存器)， 指向主存的某条机器语言指令

> ---
>
> **存储设备层次结构**
>
> - L0. 寄存器
> - L1高速缓存 SRAM(Static Random-Acess Memory)
> - L2 SRAM
> - L3 SRAM
> - L4: 主存/内存(DRAM)
> - L5 本地二级存储 本地磁盘
> - L6 远程二级存储 分布式文件系统、Web服务器

> ---
>
> **操作系统**
>
> - 文件 — 对I/O设备的抽象表示
> - 虚拟内存—对主存和磁盘I/O设备的抽象表示
> - 进程—对处理器、主存、I/O设备的抽象表示
>
> 进程：操作系统对一个正在运行的程序的一种抽象。进程交错运行的机制：上下文切换
>
> 线程：共享同样的代码和全局数据，多线程能够更高效
>
> 虚拟内存：为进程提供一个假象：每个进程都在独占使用主存
>
> > 虚拟地址空间(逐步向上)：
> >
> > 程序代码和数据
> >
> > 堆 malloc和free可以动态扩展收缩
> >
> > 共享库 存放标准库→动态链接
> >
> > 栈 编译器用它来实现函数调用，调用，栈增长，返回，栈收缩。
> >
> > 内核虚拟内存： 必须调用内核才能执行一些操作

> **Amdahl定律**
>
> $T_{new} = (1-\alpha)T_{old}+(\alpha T_{old})/k = T_{old}[(1-\alpha +\alpha/k)]$ 
>
> 加速比S=$T_{old}/T_{new} = 1/[(1-\alpha)+\alpha/k]$
>
> 想要显著加速整个系统，必须提升全系统中大部分的速度

> **并发与并行**
>
> 并发Concurrency  一个同时具有多个活动的系统
>
> 并行Parallelism 用并发使一个系统运行得更快
>
> 1. 线程级并发
>
>    多线程 指 允许一个CPU执行多个控制流的技术
>
> 2. 指令级并行
>
> 3. 单指令、多数据并行

#### ch2 信息的表示和处理

> **无符号编码、补码、浮点数**
>
> > C语言的指针的值都是某个存储块的第一个字节的虚拟地址
>
> 字长，指明指针数据的标称大小，，决定了虚拟地址空间的最大大小
>
> 64位机器 虚拟地址范围：$0 ～ 2^{64}-1$ 

> ---
>
> **寻址和字节顺序**
>
> 小端法：最低有效字节排在最前面 ✓
>
> 大端法：最高有效字节排在最前面

> ---
>
> 位运算：1，0
>
> 逻辑运算的本质：非零参数表示True/1， 参数0表示False/0
>
> 另一个却别，对第一个参数求职能确定表达式结果，就不会对第二个参数求值

> 移位
>
> 左移k位，丢弃最高的k位，右端补k个0
>
> 逻辑右移 左端补k个0
>
> 算数右移k位，左端补k个最高有效位的值，得到的结果是
>
> 对有符号数的操作是算数右移，对无符号数必须逻辑右移

> ---
>
> **补码**
>
> $-x_{\omega-1}2^{w-1}+\Sigma_{i=0}^{\omega-2}x_i2^i$ 即最高位为符号位，1为负数，0为非负数
>
> -1这个数的补码和无符号数的最大值有相同的位表示
>
> 补码和无符号数的关系：负数 = $x+2^\omega$ 非负数=$x$  

> ---
>
> **整数运算**
>
> 溢出的时候进行截断 $-2^\omega$
>
> 无符号数取反：$-x = 2^{\omega} - x$
>
> 补码加法、

> ---
>
> **浮点数**
>
> $m = x×2^y$      0.5可以精确表示  $V=(-1)^s×M×2^E$
>
> 单精度： s(符号位) 8位乘数[可能为负] 23位为系数部分
>
> 双精度  s(符号位) 11位乘数部分52位系数
>
> 规格化数【exp位不为0或0xff】：M = 1+f
>
> 非规格化数 M = f
>
> 向偶数舍入，即将结果最低有效位位偶数



#### ch3 程序的机器及表示

> 处理器状态：
>
> - 程序计数器(PC,%rip) 给出将要执行的下一条指令在内存中的地址
> - 整数寄存器文件 包含16个命名的位置，分别存储64位的值
> - 条件码寄存器 保存最近执行的算数或逻辑指令的状态信息，如if、 while
> - 一组向量寄存器 存放一个或多个整数或浮点数值
>
> `gcc -S tmp.c` 汇编语言
>
>  `gcc -c tmp.c` 二进制
>
>  `objdump -d tmp.o`  反汇编器获得字节码
>
> movb/w/l/q/absq 字节、字、双字、四字、绝对四字
>
> 栈：栈向低地址方向增长，进栈是减少栈指针(%rsp)

---

> **过程**
>
> 函数、方法、子例程、处理函数
>
> - 传递控制：P调用Q，进入Q时，程序计数器必须背设置为Q的代码其实地址，然后返回时，要把程序计数器设置为P调用Q后面一条指令的地址
> - 传递数据：P必须向Q提供一个或多个参数，Q必须能够向P返回一个值
> - 分配和释放内存：在开始时，Q可能要为局部变量分配空间，而在返回前，必须释放这些存储空间
>
> **运行时的栈**
>
> C使用“栈”数据结构提供的后进先出的内存管理原则
>
> 栈桢： 栈顶：参数构造区→局部变量→被保存的寄存器【函数Q】→返回地址→参数123【调用函数P的桢】→较早的桢 栈底
>
> **转移控制**
>
> PC设置为Q代码的起始位置，返回时记录好继续P的执行代码位置
>
> 将地址A(返回地址)压入栈中 ret会从栈中弹出A，并将PC设为A
>
> **数据传送**
>
> 参数：通过%rdi,%rsi等寄存器实现； 返回值:%rax

#### ch4 处理器体系结构

> 一个处理器支持的指令和指令的字节级编码成为他的指令集体系结构(Instruction-Set Architecture, ISA)
>
> **Y86-64**
>
> 15个程序寄存器、寄存器%rsp栈指针、条件码ZF、SF、OF，程序计数器 PC、DMEM内存，Stat 状态码[AOK, HLT, ADR非法地址≤,INS非法指令]
>
> Y86是x86的一个子集
>
> 只支持8字节整数操作
>
> - movq -> irmovq, rrmovq, mrmovq, rmmovq
>
>   立即数i, 寄存器r, 内存m
>
>   第一个为源，第二个为目的
>
> - 整数操作指令 addq, subq, andq, xorq 
>
>   ZF, SF, OF （零，符号，溢出）
>
> - 跳转指令 jmp, jle, jl, je, jne, jge, jg
>
>   <img src="/Users/cxw/Learn/2_SIGS/ALi/1_Pre/image-20210106174021991.png" alt="image-20210106174021991" style="zoom:50%;" />
>
> - 条件传送指令 cmovle, cmovl, cmove, cmovne, cmovge, cmovg
>
> - call 指令返回地址入栈，跳到目的地址，ret从调用中返回
>
> - pushq popq入栈和出栈
>
> - halt 停止指令执行, 状态码将被设为HLT
>
> 指令的字节级编码，高四位代码部分，低四位为功能部分

---

#### chp5 优化程序性能

> 几点：1. 适当的算法和数据结构；2.编译器能够有效优化转换成高效可执行代码
>
> 程序优化：1.消除不必要的工作；2. 利用处理器理工的指令级并行能力，同时执行多条指令
>
> 1. 优化编译器的能力和局限性
>
>    编译器具有优化级别`-0g`基本优化, `-02`成为被接受标准
>
> 2. 表示程序性能
>
>    度量标准：每元素的周期数(Cycles Per Element, CPE)
>
>    处理器活动顺序由始终控制，提供某频率规律信号 GHz [10周期/s] 4GHz → 0.25ns
>    
> 3. 消除循环的低效率
>
>    在循环中，避免判断或计算
>
> 4. 减少过程调用
>
> 5. 消除不必要的内存引用

#### chp6 存储器层次结构

> - CPU寄存器
>
>   0-1个时钟周期就能访问到
>
> - 高速缓存
>
>   4-75个周期
>
> - 主存储器
>
>   上百个周期
>
> - 磁盘
>
>   几千万个周期
>
> 1.1 随机访问存储器(Random Access Memory, RAM)
>
> > SRAM快一点，更贵，多用于高速缓存<=几M，而DRAM多用于主存
> >
> > 1. 静态RAM(SRAM)
> >
> >    双稳态 六晶体管电路，只要有电，就会永远保持自己的值
> >
> > 2. 动态RAM(DRAM)
> >
> >    对干扰敏感，典雅被扰乱后们就不会再恢复，需要刷新
> >
> > 3. 传统DRAM
> >
> >    先读行，再读行中某个元素
> >
> > 4. 内存模块
> >
> > 5. 增强的DRAM
> >
> >    1. 快页模式：对同一行的连续访问无需多次RAS
> >    2. 扩展数据输出DRAM
> >
> > 6. 非易失性存储器
> >
> >    易失的 volatile
> >
> >    PROM只能被变成一次，然后熔断
> >
> >    EPROM可擦写可编程，EEPROM电子可擦除
> >
> >    闪存 基于EPPROM
> >
> > 7. 访问主存
>
> 1.2. 磁盘存储
>
> 读信息是毫秒级别
>
> 1.3 固态硬盘 (基于闪存)
>
> 2. 局部性
>
>    1. 对程序数据引用的局部性 [二维数组]
>    2. 取指令的局部性
>
> 3. 存储器层次结构
>
>    | L0:寄存器             |
>    | --------------------- |
>    | L1:高速缓存 SRAM      |
>    | L2 高速缓存SRAM       |
>    | L3 高速缓存SRAM       |
>    | L4 储存DRAM           |
>    | L5 本次二级存储(磁盘) |
>    | L6 远程二级存储       |
>
>    1. - 缓存命中：当前层存在需要的数据对象
>       - 缓存不命中：不存在
>       - 缓存不命中种类：空的缓存：冷缓存
>       - 缓存管理

#### chp7 链接

> 2. 静态链接
>
>    LD命令
>
>    1. 符号解析
>    2. 重定位
>
> 3. 目标文件
>
>    - 可重定位目标文件 
>
>      包含二进制代码和数据，可以在编译时和其他可重定位目标文件合并，创建一个可执行目标文件
>
>    - 可执行目标文件
>
>      形式：可直接被复制到内存并执行
>
>    - 共享目标文件
>
>      特殊类型的可重定位目标文件，可以在加载或运行时被动态加载进内存并链接
>
> 4. 可重定位目标文件
>
>    - .text:已编译程序的机器代码
>    - .rodata: 只读数据，如printf的格式串
>    - .data: 已初始化的全局和静态变量，局部变量保存在栈中
>    - .bss: 未初始化的全局和静态变量和所有被初始化为0的
>    - .symtab: 一个符号表，存放函数和全局变量信息
>    - .rel .text 一个.text节中位置的列表
>    - .rel .data: 被模块引用或定义的所有全局变量的重定位信息
>    - .debug: 一个调试符号表
>    - .line 原始C源程序的行号和.text节中机器指令的映射
>    - .strtab 一个字符串表

#### chp11 网络编程

> C/S
>
> 基本操作是“事务”(transaction),包括4步
>
> 1. 客户端发送请求，发一个事务
> 2. 收到请求，解释它，并以适当方式操作它的资源。
> 3. 服务器向客户端发送一个响应，并等待下一个请求
> 4. 客户端收到响应并处理它
>
> ---
>
> 2. 网络
>
> LAN(局域网)[最流行的以太网技术:以太网]→
>
> 集线器 → 网桥(桥接以太网) → 路由器(不兼容的局域网)
>
> WAN(广域网) 通过路由器将多个局域网互联

#### chp12 并发编程

---

####  常识

> **Unix、Posix和标准Unix**
>
> Posix是一个标准，包含如Unix调用的C语言接口、Shell程序和工具、线程及网络编程。
>
> 这类标准化工作使Unix版本之间的差异基本消失。



## 算法

### 数组

### 链表

### 树

#### 二叉树

```
Node{
	value: any;
	left: Node | null;
	right: Node | null;
}
```

树的遍历



迭代：三色标记法 白色表示尚未访问；灰色表示尚未完全访问；黑色表示子节点全部访问

二叉树中序遍历的迭代式写法

```python
class Solution:
   def inorderTraversal(self, root:TreeNode) -> List[int]:
      WHITE, GRAY = 0, 1
      res = []
      stack = [(WHITE, root)]
      while stack:
        color, node = stack.pop()
        if node is None: continue
        if color == WHITE:
          stack.append((WHITE, node.right))
          stack.append((GRAY, node))
          stack.append((WHITE, node.right))
        else:
          res.append(node.val)
      return res
```

DFS



BFS 用队列来存储



```python
class Solution:
  # 标记了层的写法
  def bfs(k):
    
    queue = collections.deque([root])
    steps = 0
    ans = []
    while queue:
      size = len(queue)
      for _ in range(size):
        node = queue.popleft()
        if(step = k) ans.append(node)
        if node.right:
          queue.append(node.right)
        if node.left:
          queue.append(node.left)
      steps += 1
    return ans
```



平衡二叉搜索树的搜索复杂度是O(logN)

数组添加和删除的时间是O(N)

### 递归与动态规划

递归中存在大量的重复计算，所以要使用记忆化递归

#### 动态规划

1. 状态转移方程
2. 临界条件
3. 枚举状态



最长回文字符串

定义：两头相等后，子串是回文串

dp[i, j]

```python
dp[i][j] = (s[i]==s[j]) and dp[i+1][j-1]

s[i]==s[j] and j-i < 3 dp[i][j] = True
```

边界:  (j-1) - (i+1) <2巧



```python
回溯
void backtracking():
  if (终止):
    收集结果
    return
  for (选择)
  	当前选择 # 处理节点
    backtracking(下一层)  # 递归函数
  	当前这个path选择要弹出  # 回溯
    
```





XGBOOST 高级决策树 [XGBoost的原理、公式推导、Python实现和应用 - 知乎](https://zhuanlan.zhihu.com/p/162001079) 



决策树集成模型  GBDT



