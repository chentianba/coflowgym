## CoflowSim架构

### 1. CoflowBenchmarkTraceProducer

**jobs**：JobCollection类型，用于存储Job，每个Job中包含若干个MapTask和ReduceTask，MapTask中记录了Job、开始时间、Machine（Rack）信息，ReduceTask还包含了ShuffleBytes信息（从Mapper总共的接收的字节数），共有526个Job

**prepareTrace()**：从文件中读取作业信息  
**getNumRacks()**：共有150个机架  
**getMachinesPerRack()**：每个Rack（机架）中有一台机器

### 2. Simulator

**NUM_RACKS**: =150,机架的数量  
**MACHINES_PER_RACK**: 每个机架中机器的数量，默认为20，在CoflowBenchmark中是1  
**jobs**: 仿真的作业集合  
**flowsInRacks**: 在每个机架中要仿真的流集合，未填充  
**activeJobs**：记录活动的Job，在`uponJobAdmission`中添加  
**sharingAlgo**：调度算法  
**isOffline**：？？？  
**considerDeadline**：考虑截止时间，默认在**FlowSimulator**中才考虑  
**deadlineMultRandomFactor**：？？？  
**CURRENT_TIME**：当前时间，单位为毫秒，初始值为0，以`EPOCH_IN_MILLIS`的粒度递增  
**numActiveTasks**：活动任务的数量，初始值为0，在CoflowSimulator的`uponJobAdmission`和`proceedFlowsInAllRacks`中被修改


**initialize(TraceProducer)**: 根据Trace安排各机架的流,重新安排各个Task  
**simulate(int)**：按照给定的时间戳进行仿真,时间戳为10×1024；分为三个过程，首先，遍历Job集合，找到可以被调度的活跃Job，并调用`uponJobAdmission`加入到`activeJobs`，其次，当新加入的Job数量不为零时，调动`afterJobAdmission`，最后，以**8毫秒**的粒度调度活动的Job（`onSchedule`），并将完成的Job移除（`afterJobDeparture`）

```
（simulate）
1. 首先找到需要被调度的Job：开始时间在调度时间段内、是否忽略（用于调试）、准入（用于截止时间）
（uponJobAdmission）
2. 对需要调度Job，加入到activeJobs和sortedJobs中，加入sortedJobs时，按照指定的Coflow规则对Job排序
（afterJobAdmission中的layoutFlowsInJobOrder）
3. 将sortedJobs中所有的流按照接收节点放置
（afterJobAdmission中的updateRatesDynamicAlpha）
4. 对sortedJobs中的每个Job
    4.1 计算在该当前带宽下，调度该Job最多需要多少时间（秒），如果当前带宽下不能调度该Job，则加入到skippedJobs
    4.2 计算在该作业调度时间内，为每个流分配多少带宽
5. 实现工作保持
    5.1 将skippedJobs中所有的流，在每个端主机上按照流的粒度公平分配带宽
    5.2 按照到达时间，对sortedJobs中的流再次分配
（onSchedule）
6. 对将调度的每条流按照计算好的带宽，更新对应Flow、Task、Job字节数
```

### 3. CoflowSimulator

**sortedJobs**：类型为`Vector<Job>`  

**uponJobAdmission(Job)**：更新作业中ReduceTask的开始时间，并按照调度规则（FIFO、SCF、SEBF等）将作业加入到`activeJobs`中  
**afterJobAdmission(long curTime)**：首先在`layoutFlowsInJobOrder`中，将机架中的流按照有序Job的顺序依次添加，其次，在`updateRatesDynamicAlpha`中根据`sortedJobs`更新Job速率  
**onSchedule(long curTime)**：改变一个`SIMULATION_QUANTA`内各机架流的状态  
**afterJobDeparture(long curTime)**：调用`updateRatesDynamicAlpha`更新Job的分配速率

**initialize(TraceProducer)**：根据Trace初始化各机架的流，并设置各个机架的空闲速率  
**calcAlphaOnline(Job, double[], double[])**：计算该Job在当前网络环境中的传输时间，单位是秒  
**layoutFlowsInJobOrder()**：将排序后`sortedJobs`的Job集合中的所有流，加入到数组`flowsInRacks`中  
**addToSortedJobs(Job)**：按照设置的调度算法，将Job放入到有序数组`sortedJobs`中对应的调度位置  
**updateRatesDynamicAlphaOneJob(Job, double, double[], double[], boolean)**：计算为Job分配实际可以使用的带宽
**updateRatesFairShare(Vector&lt;Job&gt;, double[], double[])**：使用带宽公平的方式为`skippedJobs`中的流分配带宽  
**updateRatesDynamicAlpha(final long, boolean)**：首先将所有机架的空闲带宽设置为最大带宽1Gbps,接着，重新计算速率，依次处理`sortedJobs`中的Job：将所有ReduceTask的流速率设为0，计算该Job的传输时间（比例因子），并进一步为该Job分配空闲的带宽；然后实现工作保持：使用流公平为`skippedJobs`中的流分配带宽，再根据流截止期限分配剩余空闲带宽  
**proceedFlowsInAllRacks(long curTime, long quantaSize)**：计算每个机架中的流，改变其状态（大小）

### 4. ReduceTask

**shuffleBytes**：传输的数据量，以字节数为单位

**createFlows()**：创建流集合  
**startTask(long)**：设置Task和对应的Job的开始时间

### 5. DarkCoflowSimulator

**JOB_SIZE_MULT**：=10.0，优先级队列阈值按照指数增长，其指数基数为10.0，第i个阈值的值为`INIT_QUEUE_LIMIT*10^(i-1)`，比如，第三级队列和第四级队列的阈值为`INIT_QUEUE_LIMIT*10^2`  
**NUM_JOB_QUEUES**：=10，多级优先队列的数量  
**INIT_QUEUE_LIMIT**：=10MB，优先级队列阈值的初始值
**sortedJobs**：类型为`Vector<Job>[]`  

**updateRates(long)**：为每一级的队列中的Job分配带宽，由于每一级的Job添加是按照开始时间排序过的，所以每一级的Job遵循FIFO的调度规则  
**layoutFlowsInJobOrder()**：将每个队列中Job的ReduceTask的流加入`flowsInRacks`中  
**updateJobOrder()**：从0号队列开始，根据Job的已发送字节数，对Job的优先级逐渐下降
**addToSortedJobs(Job)**：加入到0号队列，并将Job的队列号设置为0  

**afterJobAdmission(long curTime)**：首先调用`updateJobOrder()`按照**优先队列**的规则更新调度顺序，接着调用`layoutFlowsInJobOrder`将Job的流分成各个机架可发送的流，最后调用`updateRates`更新为各条流分配速率


### 6. Constants
**SIMULATION_QUANTA**：=8，传输1MB需要的时间，单位是毫秒  
**SIMULATION_SECOND_MILLIS**：=1024，一秒等于1024毫秒  
**RACK_BYTES_PER_SEC**：带宽为128Mbps  
**RACK_BITS_PER_SEC**：带宽为1Gbps  

## CoflowGym

CoflowGym设计上的考虑有：
1. coflowsim整体结构包括两部分：作业准备和作业调度，其中作业调度是一个循环过程，每次循环分为找出当前时间段上活跃的Job集合和对该Job集进行细粒度调度。在CoflowGym的设计中，step过程就是实现coflowsim作业调度的一次循环，然后通过不断地step，实现作业的调度过程。
2. step的设计：从总体上看，gym通过对Job集合的调度状态，调节多级反馈队列MLFQ的阈值；因此，在step中，首先生成当前的活动作业集合，然后生成当前的状态，根据状态生成对应的动作action，动作生效过程中调节MLFQ阈值，最后进行实际的Coflow调度，（调度完成后生成活动Job集合）。

### state format
Coflow大小：coflow中包含的全部流的已发送字节数的总和；
Coflow流总数目：coflow包含的流数目（活跃流、完成流）；
Coflow活跃流数目：coflow当前时刻的活跃流数目；
Coflow持续时间：从coflow中第一条流的开始时间到当前时间的间隔；

MLFQ的阈值的作用是根据Coflow大小将Coflow划分为不同的级别
1. Coflow标识：不用，标识号和Coflow的大小无关
2. Coflow已发送字节数：和Coflow大小直接相关
3. Coflow持续时间：coflow持续时间和coflow大小是正相关的，一般来说，coflow越大，coflow发送时间就越长，持续时间越长
4. Coflow宽度：实验中的大部分coflow是多对多模式，因此coflow宽度也和coflow的大小相关

> **状态的设计**为（标识、宽度、已发送字节数、持续时间）

### action format

> **动作的设计**为（MLFQ的阈值）

## reward design

奖励函数的设计方向为：使得Coflow的完成时间朝着降低的方向发展（DARK：2.4247392E7，SEBF：1.5005968E7）

参考AuTO的设计：使用相邻step完成流的平均吞吐量之比。因此，考虑使用：

稀疏奖励值：在一个完整trace的情况下，获得数据中coflow的总共完成时间，将该时间作为奖励值  

    这种设计将数据中coflow的总共完成时间直接作为奖励值，最为直接的驱动Agent训练

非稀疏奖励：相邻step的已完成Coflow平均完成时间之比  
    
    这种设计使用了“已完成coflow平均完成时间”作为评价指标，相邻step的“已完成coflow平均完成时间”之比越小，在对相同coflow的调度情况下，就会优先调度更短的coflow，从而coflow的平均完成时间也就越短。


> **奖励函数的设计**为相邻step的已完成Coflow平均完成时间之比

### algorithm design

由于动作的设计是离散型，因此适合使用DDPG、PPO这类算法

直接使用DDPG算法很难有效果，因此考虑使用更好的DDPG算法设计

当前DDPG使用的技术点：
* Actor-Critic网络架构：actor是策略网络用于产生动作，critic是价值网络用于估计状态动作的回报
* 使用bootstrap方法对状态进行评估：采样效率高，能使用离线数据进行训练
* 使用目标网络，目标网络更新使用软更新：使得策略网络和价值网络更加稳定
* 使用经验回放存储历史经验：采样batch训练，使得训练更加稳定

需要考虑的优化点包括：
* 神经网络优化，使用CNN、RNN等网络
* 使用优势函数优化
* 使用multi-step方法代替bootstrap
* 输入标准化
* 神经网络梯度下降优化方法
* 神经网络学习因子调节
* 添加置信区间优化——TRPO、PPO
* clip奖励值

**DRL的训练优化**：目的是降低DRL的训练难度
对真正意义上的网络调度问题，不像游戏那样存在显示的episode概念，从游戏的开始到游戏成功/失败的过程可以为一个episode，网络调度问题只存在网络环境的建立以及无穷的网络调度，网络调度不存在成功和失败；而在传统的RL模型中，episode是一个很重要的概念，
1. 经验回放池优化：
    * Priority经验回放（出）：通过对经验样本的TD误差设置优先级，实现从经验回放池中淘汰旧样本
    * 样本数据分类（入）：好的动作探索环境产生好的样本数据，差的动作产生差的样本数据，通过对生成的动作进行人工干预，使得训练朝好的方向优化
2. 终止条件优化：
    * 设置动态终止条件：完整的Benchmark回放，即一个episode需要的时间超过5min，中间包括500个左右的step，也就是500个动作(连续)选择，这意味着：第一，如果前期出现错误的动作，需要几百个step才能得到反馈，从而需要更长的训练周期；第二，更长的马尔科夫决策链使得探索的状态空间更大，适当的缩小状态空间有助于减小状态空间探索难度。

## 算法调试

问题一：输出的动作是极大值或者极小值
1. 输入没有标准化、归一化，各维度数值数量级差异较大
2. 激活函数使用tanh，网络过深导致tanh求导为0,而无法学习，减少Actor、Critic的隐藏层

参照[神经网络的数值问题](https://www.jianshu.com/p/95a8f035c86c)和[MVFST-RL](https://arxiv.xilesou.top/pdf/1910.04054.pdf)的标准化方式.

> 归一化
* 线性变换和极差法（线性归一化）  
该方法实现对原始数据的等比例缩放。通过利用变量取值的最大值和最小值（或者最大值）将原始数据转换为界于某一特定范围的数据，从而消除量纲和数量级影响，改变变量在分析中的权重来解决不同度量的问题。由于极值化方法在对变量无量纲化过程中仅仅与该变量的最大值和最小值这两个极端值有关，而与其他取值无关，这使得该方法在改变各变量权重时过分依赖两个极端取值。
* 标准化方法 
每一变量值与其平均值之差除以该变量的标准差,无量纲化的同时还消除了各变量在变异程度上的差异。
> 机器学习归一化方法在DRL中存在的问题：ML中可以提前获得全部训练数据，从而按照统一标准（极大值/极小值）进行归一化，但是RL无法提前获得全部数据，只能从少到多的增加数据,因此使用线性归一化方法，并在样本获取中动态调整最大值、最小值

> 激活函数的选取：神经网络中线性模型的表达能力不够，使用激活函数加入非线性因素  
tanh函数：在特征相差明显时效果会很好，在循环过程中会不断扩大特征效果；
sigmoid函数：在特征相差复杂或是相差不是特别大时，需要更细微的分类判断时，sigmoid效果好；  
需要注意的是，sigmoid和tanh作为激活函数时，要对input进行归一化，否则激活后的值会进入平坦区，使隐层的输出全部趋同；  
ReLU函数：构建稀疏矩阵，也就是稀疏性，也就是大多数为0的稀疏矩阵来表示，可以去除数据中的冗余，最大可能保留数据的特征，这种输入适合ReLU函数；ReLU取max(0,x)，神经网络变成不断试探如何用一个大多数为0的矩阵来表达数据特征，结果因为稀疏性的存在，反而这种方法变得运算快、效果又好。缺点是会使一些神经元永久失活，在使用大的学习率时，会出现几乎所有神经元失活，输出为0的情况  
ReLU不需要输入归一化来防止达到饱和，卷积神经网络大多使用ReLU函数。
* [如何选取激活函数](https://blog.csdn.net/kongxp_1/article/details/80726409)

问题二：当前一个episode是用完整的benchmark回放，训练效率低。
原因：在训练周期里没有引入前期的训练评价机制，应当采取训练评价，动态增加episode。

总结1：折扣因子gamma和episode的长度息息相关。折扣因子是眼前利益和长远利益的权衡因子，gamma越大，agent越重视长远利益，episode靠后状态对episode靠前的状态影响就越大，需要更长的训练次数才能使神经网络agent适应未来的回报，考虑到神经网络、DRL的不稳定，训练难度就越大；同时，episode越长，神经网络收敛的难度就越大。

> 问题三：agent在探索期间，探索不到好的动作，导致长时间训练无法收敛，收敛时往往收敛到很差的结果
实验：将这种算法的参数配置运用到MountainCar-continuous-v0的环境中，通过对比他人成功的案例，发现关键点是噪声的设置
原因：



补充阅读：  
* [时延敏感网络](https://www.sdnlab.com/23891.html)  
看看流量整形和优先级队列
* [简单解释: 分布式快照(Chandy-Lamport算法)](https://www.jianshu.com/p/53be93a5e5cb?from=singlemessage)  
看看这个分布式算法，有什么启发，用在网络分布式状态获取上

## DDPG调参资料
1. [(DDPG)深度确定策略梯度调参体会](https://blog.csdn.net/qq_32231743/article/details/73770055)
2. [DDPG在连续运动控制中的一点记录](http://blog.sina.com.cn/s/blog_59dd8a360102x7k6.html)
3. [基于DDPG的TORCS自动驾驶训练笔记(二)](https://zhuanlan.zhihu.com/p/57755078)：其中的episode终止条件值得参考
4. [深度确定性策略梯度算法，越训练效果越差？](https://www.zhihu.com/question/61035679/answer/183232526)：对动作进行人工评判，分类存储在不同的buffer中，在好的环境中就能学到好动作，坏环境中学到坏动作
5. [深度强化学习落地方法论（7）—— 训练篇](https://zhuanlan.zhihu.com/p/99901400)：数据预处理reward rescale、折扣因子的取值原则是在算法能够收敛的前提下尽可能大、agent“看得远”表面上指的是向前考虑的步数多实质上是指agent向前考虑的系统动态演化跨度大；如果之前的经验对当前决策很有参考意义（比如Dota）就适合用RNN，反之仅依靠即时信息做应激式决策就足以应付就没必要用RNN。实践中经常采取折中方案，将最近几个step的原始状态信息叠加到一起作为当前时刻的实际状态信息输入policy，既可以挖掘一定范围内的时序信息，又避免增加训练难度。
6. [Soft Actor-Critic算法](https://zhuanlan.zhihu.com/p/70360272)

重要资料：
1. [深度强化学习落地方法论](https://zhuanlan.zhihu.com/c_1186982555915599872)