# [BroadFace](https://arxiv.org/abs/2008.06674)
#### 收录到[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)

# 说明
- ECCV2020
- 2020.11.27  学习率倍增
- 2020.11.25  开源
- 2020.9.11    第三方实现

## 算法框架
适用场景
> 微调：加载预训练权重。否则队列内特征误差过大，导致震荡

实现原理
> 1. 通过 当前类中心与过去类中心的差值，将 队列内过去特征 补偿近似为 当前特征, 实现 大Batch更新分类器。
> 2. 更好的分类器训练更好的卷积层
![](https://github.com/bobo0810/BroadFace/blob/main/imgs/broadface.png)


## 环境

| python版本 | pytorch版本 | 系统   |
|------------|-------------|--------|
| 3.6        | 1.6.0       | Ubuntu |

