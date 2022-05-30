# jonily_zhang
# 导入需要的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D# 三维数据的可视化
import hiddenlayer as hl
from sklearn.manifold import TSNE
from sklearn.svm import SVC# 建立支持向量机分类器
from sklearn.decomposition import PCA# 对数据进行主成分分析以获取数据的主成分
from sklearn.metrics import classification_report,accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

# 使用手写体数据，准备训练数据集
train_data = MNIST(
    root="./data/MNIST",
    train = True,
    transform = transforms.ToTensor(),
    download=True)
# 将图像数据转换成向量数据
train_data_x = train_data.data.type(torch.FloatTensor)/255.0
train_data_x = train_data_x.reshape(train_data_x.shape[0],-1)#图像数据
train_data_y = train_data.targets# 标签数据
# 定义一个数据加载器
train_loader = Data.DataLoader(
    dataset = train_data_x,
    batch_size = 64,
    shuffle = True,
    num_workers = 2,)
# 对训练数据集进行导入
test_data = MNIST(
    root="./data/MNIST",
    train=False,# 只使用训练数据集
    transform = transforms.ToTensor(),
    download = True)
# 为测试数据添加一个通道维度，获取测试数据的X和Y
test_data_x = test_data.data.type(torch.FloatTensor)/255.0
test_data_x = test_data_x.reshape(test_data_x.shape[0],-1)
test_data_y = test_data.targets
print("train_data:",train_data_x.shape)
print("test_data:",test_data_x.shape)

# 可视化一个batch的图像内容，获得一个batch的数据
for step, b_x in enumerate(train_loader):
    if step>0:
        break
# 可视化一个batch的图像
im = make_grid(b_x.reshape((-1,1,28,28)))
# make_grid()是Pytorch库中的函数，可以直接将数据结构[batch,channel,height,width]形式的batch图像转换为图像矩阵
im = im.data.numpy().transpose((1,2,0))# 交换矩阵数据位置，先1维与2维换，再2维与3维换
plt.figure()
plt.imshow(im)
plt.axis("off")# 关闭坐标轴
plt.show()

# 构建一个EnDecoder()类
class EnDecoder(nn.Module):
    def __init__(self):
        super(EnDecoder,self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(784,512),
            nn.Tanh(),# 双曲正切激活函数
            nn.Linear(512,256),
#      nn.Linear()是用于设置网络中的全连接层的，需要注意在二维图像处理的任务中，
#     全连接层的输入与输出一般都设置为二维张量，形状通常为[batch_size, size]，不同于卷积层要求输入输出是四维张量。
#     从输入输出的张量的shape角度来理解,相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
            nn.Linear(128,3),
            nn.Tanh(),)
        self.Decoder = nn.Sequential(
            nn.Linear(3,128),
            nn.Tanh(),
            nn.Linear(128,256),
            nn.Tanh(),
            nn.Linear(256,512),
            nn.Tanh(),
            nn.Linear(512,784),
            nn.Sigmoid(),)
    def forward(self,x):
            encoder = self.Encoder(x)
            decoder = self.Decoder(encoder)
            return encoder,decoder
          
# 输出网络结构
edmodel = EnDecoder()
print(edmodel)

# 定义优化器
optimizer  =torch.optim.Adam(edmodel.parameters(),lr=0.003)
loss_func = nn.MSELoss()
# 记录训练过程的指标
history1 = hl.History()
# 使用canvas进行可视化
canvas1 = hl.Canvas()
train_num = 0
val_num = 0
for epoch in range(10):
    train_loss_epoch = 0
    for step, b_x in enumerate(train_loader):
        _,output = edmodel(b_x)# 在训练batch上的输出
        loss = loss_func(output,b_x)# 均方根误差
        optimizer.zero_grad()# 每个迭代步的梯度初始化为0
        loss.backward()# 损失的向后传播，计算梯度（也即对y求导）
        optimizer.step()# 使用梯度进行优化
        train_loss_epoch+=loss.item()*b_x.size(0)
        train_num = train_num + b_x.size(0)
#     计算每一个epoch的损失
    train_loss = train_loss_epoch/train_num
#     保存每个epoch上的输出loss
    history1.log(epoch,train_loss=train_loss)
#     可视网络训练过程
    with canvas1:
        canvas1.draw_plot(history1["train_loss"])
      
# 预测测试集前100张图像的输出
edmodel.eval()# 将模型设置成验证模式
_,test_decoder = edmodel(test_data_x[0:100,:])
# 可视化原始图像
plt.figure(figsize=(6,6))
for ii in range(test_decoder.shape[0]):
    plt.subplot(10,10,ii+1)
    im = test_data_x[ii,:]
    im = im.data.numpy().reshape(28,28)
    plt.imshow(im,cmap=plt.cm.gray)
    plt.axis("off")
plt.show()
# 可视化编码后的图像
plt.figure(figsize=(6,6))
for ii in range(test_decoder.shape[0]):
    plt.subplot(10,10,ii+1)
    im = test_decoder[ii,:]
    im = im.data.numpy().reshape(28,28)
    plt.imshow(im,cmap=plt.cm.gray)
    plt.axis("off")
plt.show()

# 获取前500个样本的自编码后的特征，并对数据进行可视化
edmodel.eval()
TEST_num = 500
test_encoder, _ = edmodel(test_data_x[0:TEST_num,:])
print("test_encoder.shape:",test_encoder.shape)

# 二维展示
test_encoder_arr = test_encoder.data.numpy()
X = test_encoder_arr[:,0]
Y=test_encoder_arr[:,1]
plt.figure(figsize=(8,6))
plt.xlim([min(X)-0.1,max(X)+0.1])
plt.ylim([min(Y)-0.1,max(Y)+0.1])
for ii in range(test_encoder.shape[0]):
    text = test_data_y.data.numpy()[ii]
    plt.text(X[ii],Y[ii],str(text),fontsize=8,bbox = dict(boxstyle="round",facecolor=plt.cm.Set1(text),alpha = 0.7))
plt.show()

%config InlineBackend.print_figure_kwargs = {'bbox_inches':None}
# 将3个维度的特征进行可视化
test_encoder_arr = test_encoder.data.numpy()
fig = plt.figure(figsize=(12,8))
ax1 = Axes3D(fig)
X = test_encoder_arr[:,0]# numpy数组中的一个写法，[:,0]表示的是取所有行的第0个数据，[:,1]表示的是取所有行的第1个数据
Y = test_encoder_arr[:,1]
Z = test_encoder_arr[:,2]
ax1.set_xlim([min(X),max(X)])# min和max分别表示的是各个坐标轴的下上限
ax1.set_ylim([min(Y),max(Y)])
ax1.set_zlim([min(Z),max(Z)])
for ii in range(test_encoder.shape[0]):
    text = test_data_y.data.numpy()[ii]
    ax1.text(X[ii],Y[ii],Z[ii],str(text),fontsize=8,
            bbox = dict(boxstyle="round",facecolor = plt.cm.Set1(text),alpha=0.7))
plt.show()

# 自编码后的特征训练集和测试集
train_ed_x,_ = edmodel(train_data_x)
train_ed_x = train_ed_x.data.numpy()
train_y = train_data_y.data.numpy()
test_ed_x,_ = edmodel(test_data_x)
test_ed_x = test_ed_x.data.numpy()
test_y = test_data_y.data.numpy()

# PCA降维获得训练集和测试集前3个主成分
pcamodel = PCA(n_components = 3,random_state=10)
train_pca_x = pcamodel.fit_transform(train_data_x.data.numpy())
test_pca_x = pcamodel.transform(test_data_x.data.numpy())
print(train_pca_x.shape)

# 使用自编码数据建立分类器，训练和预测
encodersvc = SVC(kernel="rbf",random_state = 123)
encodersvc.fit(train_ed_x,train_y)
edsvc_pre = encodersvc.predict(test_ed_x)
print(classification_report(test_y,edsvc_pre))
print("model accuracy:",accuracy_score(test_y,edsvc_pre))

# 使用PCA降维后训练得到的SVM分类器
pcasvc = SVC(kernel="rbf",random_state = 123)
pcasvc.fit(train_pca_x,train_y)
pcasvc_pre = pcasvc.predict(test_pca_x)
print(classification_report(test_y,pcasvc_pre))
print("model accuracy:",accuracy_score(test_y,pcasvc_pre))

"""
一些读书笔记：
1.PCA目的/作用：
    主成分分析算法（PCA）是最常用的线性降维方法，它的目标是通过某种线性投影，将高维的数据映射到低维的空间中，
并期望在所投影的维度上数据的信息量最大（方差最大），以此使用较少的数据维度，同时保留住较多的原数据点的特性。
    PCA降维的目的，就是为了在尽量保证“信息量不丢失”的情况下，对原始特征进行降维，
也就是尽可能将原始特征往具有最大投影信息量的维度上进行投影。将原特征投影到这些维度上，使降维后信息量损失最小。
    求解步骤：
    ①去除平均值
    ②计算协方差矩阵
    ③计算协方差矩阵的特征值和特征向量
    ④将特征值排序
    ⑤保留前N个最大的特征值对应的特征向量
    ⑥将原始特征转换到上面得到的N个特征向量构建的新空间中（最后两步，实现了特征压缩）

2.Pytorch 为什么每一轮batch需要设置optimizer.zero_grad：
    根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。

3.关于python中numpy.transpose详解 参考博客：https://blog.csdn.net/u012762410/article/details/78912667

4.激活函数解析参考博客：https://blog.csdn.net/g11d111/article/details/105888269

5.numpy中的axis和Pytorch中的dim参数解析 参考博客：https://www.cnblogs.com/liujianing/p/13236448.html

6.python canvas教程_Canvas 参考博客：https://blog.csdn.net/weixin_39959349/article/details/112041338

7.损失函数解析参考博客：https://blog.csdn.net/weixin_57643648/article/details/122704657

8.关于matplotlib系列-plt.axis参数解析 参考博客：https://blog.csdn.net/jose_m/article/details/105594038
"""
