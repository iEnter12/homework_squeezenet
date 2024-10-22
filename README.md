# SqueezeNet
SqueezeNet 在 ImageNet 上实现与 AlexNet 同等级别的精度，但 参数少了50倍 ；
通过模型压缩技术，实现了 SqueezeNet 模型压缩到小于0.5MB，相比 AlexNet 模型小了510倍 ；
论文地址：[SqueezeNet](https://arxiv.org/abs/1602.07360)

## Fire Module
（1）使用 1 x 1 卷积滤波器代替 3 x 3 卷积 （参数量少9倍）；
（2）使用3x3个滤波器减少输入通道的数量，利用 squeeze layers 实现 ；
（3）在网络后期进行下采样操作，可以使卷积层有更大的激活特征图 ；
[图片1](./img/Screenshot%202024-10-22%20192739.png "图片1")

## SqueezeNet 结构
图中分别为未修改的SqueezeNet, 带简单旁路的SqueezeNet, 带复杂旁路的SqueezeNet。
[图片2](./img/Screenshot%202024-10-22%20192856.png "图片2")
