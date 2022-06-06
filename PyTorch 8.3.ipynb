# jonily_zhang
# 导入所需要的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D# 三维数据的可视化
import hiddenlayer as hl
from sklearn.model_selection import train_test_split
# 下载skimage出错解决办法，参考博客：https://blog.csdn.net/com_fang_bean/article/details/103563608
from skimage.util import random_noise
# from skimage.measure import  compare_psnr,compare_ssim# 注意在新版本舍弃了这两种方法的引用
# 然后在API文档中搜索找到了新版本应该使用的API，即执行如下的语句
# 参考博客：https://blog.csdn.net/qq_36571422/article/details/122462988
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
from sklearn.manifold import TSNE
from sklearn.svm import SVC# 建立支持向量机分类器
from sklearn.decomposition import PCA# 对数据进行主成分分析以获取数据的主成分
from sklearn.metrics import classification_report,accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision.datasets import STL10
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

# 该训练网络使用到的图像数据集——STL10，共包含三种类型数据，分别是带有标签的训练集与验证集，
# 分别包含5000张和8000张图像，共有10类数据
# 还有一个类型包含10万张的无标签图像，均是96*96的RGB图像，均可用于无监督学习

# 关于RGB：
# RGB色彩就是常说的光学三原色，R代表Red（红色），G代表Green（绿色），B代表Blue（蓝色）。
# 自然界中肉眼所能看到的任何色彩都可以由这三种色彩混合叠加而成，因此也称为加色模式。

# 定义一个将bin文件处理为图像数据的函数
def read_image(data_path):
    with open(data_path,'rb') as f:
#         with...as...将with后面的参数处理后传给f
#        具体讲解见博客：https://blog.csdn.net/zhauuu/article/details/122517057
#         https://blog.csdn.net/m0_61655732/article/details/120636660?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-120636660-blog-122517057.pc_relevant_aa&spm=1001.2101.3001.4242.2&utm_relevant_index=4
#         'rb',按照二进制位进行读取的，不会将读取的字节转换成字符
#         关于Python文件读取模式，参考博客：https://blog.csdn.net/weixin_39906245/article/details/110397609
        datal = np.fromfile(f,dtype = np.uint8)# 表示utf-8编码的字符串。参考博客:https://blog.csdn.net/qq_42191914/article/details/103103460
#         图像[数量，通道，宽，高]
        images = np.reshape(datal,(-1,3,96,96))
#     图像转换为RGB形式，方便使用matplotlib进行可视化
#     关于np.transpose解析，见博客：https://blog.csdn.net/qq_36387683/article/details/82228131
        images = np.transpose(images,(0,3,2,1))
#     最后输出的像素值是在0 ~ 1之间的四维数组，第一维表示图像的位置，后面三维表示图像的RGB像素值
#     因为是utf-8编码，所以是除255
    return images/255.0

data_path="D:/The CIFAR-10 dataset/stl10_binary/train_X.bin"
images = read_image(data_path)
print("images.shape:",images.shape)

# 为图像数据添加一个高斯噪声的函数
def gaussian_noise(images,sigma):
#     sigma:噪声标准差
    sigma2 = sigma**2/(255**2)# 噪声方差
    images_noisy = np.zeros_like(images)
    for ii in range(images.shape[0]):
        image = images[ii]
#         使用skimage库中的random_noise函数添加噪声
#         有关通过skimage.util.random_noise添加噪声，可参考博客：https://blog.csdn.net/weixin_44457013/article/details/88544918?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-88544918-blog-103325097.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-88544918-blog-103325097.pc_relevant_aa&utm_relevant_index=2
        noise_im = random_noise(image,mode="gaussian",var=sigma2,clip=True)
        images_noisy[ii] = noise_im
    return images_noisy
images_noise = gaussian_noise(images,30)
print("images_noise:",images_noise.min(),"~",images_noise.max())# 由输出可得知：所有像素值的最大值为1，最小值为0

# 可视化其中的部分图像，不带噪声的图像
plt.figure(figsize=(6,6))
for ii in np.arange(36):
    plt.subplot(6,6,ii+1)
#     关于python matplotlib在一张画布上画多个图的两种方法，plt.subplot(),plt.subplots()
#     可参考博客：https://blog.csdn.net/qq_45058254/article/details/105895130?spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-105895130-blog-112145027.pc_relevant_paycolumn_v3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-105895130-blog-112145027.pc_relevant_paycolumn_v3&utm_relevant_index=13
    plt.imshow(images[ii,...])
    plt.axis("off")
plt.show()
plt.figure(figsize=(6,6))
for ii in np.arange(36):
    plt.subplot(6,6,ii+1)
    plt.imshow(images_noise[ii,...])
    plt.axis("off")
plt.show()

# 数据准备为Pytorch可用的形式，转化为[样本，通道，高，宽]的数据形式
data_Y = np.transpose(images,(0,3,2,1))
data_X = np.transpose(images_noise,(0,3,2,1))
# 将数据集切分为训练集和验证集
# 关于train_test_split()函数，参考博客:https://blog.csdn.net/qq_39355550/article/details/82688014
X_train,X_val,y_train,y_val = train_test_split(
    data_X,data_Y,test_size=0.2,random_state=123)
# 将图像数据转化为向量数据
X_train = torch.tensor(X_train,dtype=torch.float32)
y_train = torch.tensor(y_train,dtype=torch.float32)
X_val = torch.tensor(X_val,dtype=torch.float32)
# 将X和Y转化为数据集合
train_data = Data.TensorDataset(X_train,y_train)
val_data = Data.TensorDataset(X_val,y_val)
print("X_train.shape:",X_train.shape)
print("y_train.shape:",y_train.shape)
print("X_val.shape:",X_val.shape)
print("y_val.shape:",y_val.shape)

train_loader = Data.DataLoader(
    dataset = train_data,# 使用的数据集
    batch_size=32, # 批处理样本大小
    shuffle = True,# 每次迭代前打乱数据
    num_workers = 4,) # 使用4个进程
val_loader = Data.DataLoader(
    dataset = val_data,
    batch_size=32,
    shuffle = True,
    num_workers = 4,)

# 搭建卷积自编码网络
class DenoiseAutoEncoder(nn.Module):
    def __init__(self):
#         关于python中super().__init__()，参见博客:https://blog.csdn.net/a__int__/article/details/104600972
        super(DenoiseAutoEncoder,self).__init__()
        self.Encoder = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=64,
                 kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,64,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,64,3,1,1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,128,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,128,3,1,1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,256,3,1,1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(256),
        )
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,128,3,2,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,32,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32,32,3,1,1),
            nn.ConvTranspose2d(32,16,3,2,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16,3,3,1,1),
            nn.Sigmoid(),)
    def forward(self,x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder,decoder

DAEmodel = DenoiseAutoEncoder()
print(DAEmodel)

# 定义优化器
LR = 0.0003
optimizer = torch.optim.Adam(DAEmodel.parameters(),lr=LR)
loss_func = nn.MSELoss() # 损失函数
history1 = hl.History()
canvas1 = hl.Canvas()
train_num = 0
val_num = 0
for epoch in range(10):
    train_loss_epoch=0
    val_loss_epoch=0
#     对训练数据的加载器进行迭代计算
    for step,(b_x,b_y) in enumerate(train_loader):
        DAEmodel.train()
#         使用每个batch进行训练模型
        _,output =DAEmodel(b_x)
        loss = loss_func(output,b_y)# 均方根误差
        optimizer.zero_grad()# 每个迭代步的梯度初始化为0
        loss.backward()# 损失的后向传播，计算梯度
        optimizer.step()# 使用梯度进行优化
        train_loss_epoch+=loss.item()* b_x.size(0)
        train_num = train_num+b_x.size(0)
    for step,(b_x,b_y) in enumerate(val_loader):
        DAEmodel.eval()
        _,output = DAEmodel(b_x)
        loss = loss_func(output,b_y)
        val_loss_epoch+=loss.item()*b_x.size(0)
        val_num  =val_num+b_x.size(0)
    train_loss = train_loss_epoch/train_num
    val_loss = val_loss_epoch/val_num
    history1.log(epoch,train_loss =train_loss,
                val_loss=val_loss)
    with canvas1:
        canvas1.draw_plot([history1["train_loss"],history1["val_loss"]])

imageindex = 1
im = X_val[imageindex,...]
im = im.unsqueeze(0)
imnose = np.transpose(im.data.numpy(),(0,3,2,1))
imnose = imnose[0,...]
DAEmodel.eval()
_,output = DAEmodel(im)
imde = np.transpose(output.data.numpy(),(0,3,2,1))
imde = imde[0,...]
im = y_val[imageindex,...]
imor = im.unsqueeze(0)
imor = np.transpose(imor.data.numpy(),(0,3,2,1))
imor = imor[0,...]
print("Add noisy PSNR:",compare_psnr(imor,imnose),"dB")
print("wipe off noisy PSNR:",compare_psnr(imor,imde),"dB")

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(imor)
plt.axis("off")
plt.title("Origin Image")
plt.subplot(1,3,2)
plt.imshow(imnose)
plt.axis("off")
plt.title("Noise Image $\sigma$=30")
plt.subplot(1,3,3)
plt.imshow(imde)
plt.axis("off")
plt.title("Denoise Image")
plt.show()

PSNR_val=[]
DAEmodel.eval()
for ii in range(X_val.shape[0]):
    imageindex = ii
    im = X_val[imageindex,...]
    im = im.unsqueeze(0)
    imnose = np.transpose(im.data.numpy(),(0,3,2,1))
    imnose = imnose[0,...]
    _,output = DAEmodel(im)
    imde = np.transpose(output.data.numpy(),(0,3,2,1))
    imde = imde[0,...]
    im = y_val[imageindex,...]
    imor = im.unsqueeze(0)
    imor = np.transpose(imor.data.numpy(),(0,3,2,1))
    imor = imor[0,...]
    PSNR_val.append(compare_psnr(imor,imde) - compare_psnr(imor,imnose))
print("PSNR average increase:",np.mean(PSNR_val),"dB")

class DenoiseEncoder(nn.Module):
    def __init__(self):
        super(DenoiseAutoEncoder,self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels = 64,kernel_size = 3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            )
        self.Decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),# 上采样函数
#             更多关于上采样，参考博客:https://cloud.tencent.com/developer/article/1659277
            nn.Conv2d(256,128,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,64,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64,32,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),)
    def forward(self,x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder,decoder
DAEmodel = DenoiseAutoEncoder()

LR = 0.0003
optimizer = torch.optim.Adam(DAEmodel.parameters(),lr=LR)
loss_func = nn.MSELoss()
history1 = hl.History()
canvas1 = hl.Canvas()
train_num = 0
val_num = 0
for epoch in range(10):
    train_loss_epoch=0
    val_loss_epoch=0
    for step,(b_x,b_y) in enumerate(train_loader):
        DAEmodel.train()
        _,output =DAEmodel(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch+=loss.item()* b_x.size(0)
        train_num = train_num+b_x.size(0)
    for step,(b_x,b_y) in enumerate(val_loader):
        DAEmodel.eval()
        _,output = DAEmodel(b_x)
        loss = loss_func(output,b_y)
        val_loss_epoch+=loss.item()*b_x.size(0)
        val_num  =val_num+b_x.size(0)
    train_loss = train_loss_epoch/train_num
    val_loss = val_loss_epoch/val_num
    history1.log(epoch,train_loss =train_loss,
                val_loss=val_loss)
    with canvas1:
        canvas1.draw_plot([history1["train_loss"],history1["val_loss"]])

imageindex = 1
im = X_val[imageindex,...]
im = im.unsqueeze(0)
imnose = np.transpose(im.data.numpy(),(0,3,2,1))
imnose = imnose[0,...]
DAEmodel.eval()
_,output = DAEmodel(im)
imde = np.transpose(output.data.numpy(),(0,3,2,1))
imde = imde[0,...]
im = y_val[imageindex,...]
imor = im.unsqueeze(0)
imor = np.transpose(imor.data.numpy(),(0,3,2,1))
imor = imor[0,...]
print("Add noisy PSNR:",compare_psnr(imor,imnose),"dB")
print("wipe off noisy PSNR:",compare_psnr(imor,imde),"dB")

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(imor)
plt.axis("off")
plt.title("Origin Image")
plt.subplot(1,3,2)
plt.imshow(imnose)
plt.axis("off")
plt.title("Noise Image $\sigma$=30")
plt.subplot(1,3,3)
plt.imshow(imde)
plt.axis("off")
plt.title("Denoise Image")
plt.show()

PSNR_val=[]
DAEmodel.eval()
for ii in range(X_val.shape[0]):
    imageindex = ii
    im = X_val[imageindex,...]
    im = im.unsqueeze(0)
    imnose = np.transpose(im.data.numpy(),(0,3,2,1))
    imnose = imnose[0,...]
    _,output = DAEmodel(im)
    imde = np.transpose(output.data.numpy(),(0,3,2,1))
    imde = imde[0,...]
    im = y_val[imageindex,...]
    imor = im.unsqueeze(0)
    imor = np.transpose(imor.data.numpy(),(0,3,2,1))
    imor = imor[0,...]
    PSNR_val.append(compare_psnr(imor,imde) - compare_psnr(imor,imnose))
print("PSNR average increase:",np.mean(PSNR_val),"dB")
