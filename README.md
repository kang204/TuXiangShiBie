# TuXiangShiBie
#基于python卷积神经网络模型在 CIFAR-10 数据集上进行训练和测试的图像识别系统
##因为一些原因把这个源码公开，代码是基于龙芯平台、LoongArch框架开发的
##使用的是PyTorch的Ai框架，代码只是一个简单的利用卷神经网络模型来训练图像识别系统，

## CIFAR-10 数据集已经下载好了，直接保存在data文件里，可以去跑一下，也可以直接去CIFAR-10官网下载一下这个数据集去测试
 
# windows系统因为多进程模块导致会报错，子进程无法正常启动运行 所以增加了一个解决代码
```
if __name__ == '__main__':
    train()
```
#对电脑cpu要求较大，推荐有显卡的使用 有动手能力的可以修改跑一下

**这只是一个简单的入门图像识别系统代码，需要加深的可以在源代码基础上加深**
