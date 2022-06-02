# jonily_zhang
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image

vgg16 = models.vgg16(pretrained = True)
im=Image.open("D:/train_image/n0/crocodile.jpg")
imarray = np.asarray(im)/255.0
plt.figure()
plt.imshow(imarray)
plt.show()
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
input_im = data_transforms(im).unsqueeze(0)
# print("input_im.shape:",input_im.shape)
activation= {}
def get_activation(name):
    def hook(model,input,output):
        activation[name] = output.detach()
    return hook
vgg16.eval()
vgg16.features[4].register_forward_hook(get_activation("maxpool1"))
_ = vgg16(input_im)
maxpool1 = activation["maxpool1"]
# print("获取特征的尺寸为：",maxpool1.shape)
plt.figure(figsize=(11,6))
for ii in range(maxpool1.shape[1]):
    plt.subplot(6,11,ii+1)
    plt.imshow(maxpool1.data.numpy()[0,ii,:,:],cmap="gray")
    plt.axis("off")
plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.show()
vgg16.eval()
vgg16.features[21].register_forward_hook(get_activation("layer21_conv"))
_ = vgg16(input_im)
layer21_conv = activation["layer21_conv"]
# print("the further feature shape:",layer21_conv.shape)
plt.figure(figsize=(12,6))
for ii in range(72):
    plt.subplot(6,12,ii+1)
    plt.imshow(layer21_conv.data.numpy()[0,ii,:,:],cmap="seismic")
    plt.axis("off")
plt.subplots_adjust(wspace=0.1,hspace=0.1)
# plt.show()
LABEL_PATH = 'D:/train_image/ImageNet_Labels.txt'#原本的网址失效，这里是特地找的ImageNet
with open(LABEL_PATH) as j:
    labels_orig = eval(j.read())#读取并创建对应的ImageNet—Labels字典
for i in labels_orig:#读取出来的字典是列表类型，需要再处理一下
    labels_orig[i]=str(labels_orig[i])
    labels_orig[i]=labels_orig[i][1:-1]
    labels_orig[i]=labels_orig[i].replace("'","")
vgg16.eval()
im_pre = vgg16(input_im)
softmax = nn.Softmax(dim=1)
im_pre_prob = softmax(im_pre)
prob,prelab = torch.topk(im_pre_prob,5)
prob = prob.data.numpy().flatten()
prelab = prelab.numpy().flatten()
for ii,lab in enumerate(prelab):
    print("index:",lab,"label:",labels_orig[str(lab)]," ||",prob[ii])#注意使用字典的时候，里面的要是str类型
class MyVgg16(nn.Module):
    def __init__(self):
        super(MyVgg16,self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.features_conv = self.vgg.features[:30]
        self.max_pool = self.vgg.features[30]
        self.avgpool = self.vgg.avgpool
        self.classifier = self.vgg.classifier
        self.gradients = None
    def activations_hook(self,grad):
        self.gradients = grad
    def forward(self,x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.max_pool(x)
        x= self.avgpool(x)
        x = x.view((1,-1))
        x = self.classifier(x)
        return x
    def get_activations_gradient(self):
        return self.gradients
    def get_activations(self,x):
        return self.features_conv(x)
vggcam = MyVgg16()
vggcam.eval()
im_pre = vggcam(input_im)
softmax = nn.Softmax(dim=1)
im_pre_prob = softmax(im_pre)
prob,prelab = torch.topk(im_pre_prob,5)
prob = prob.data.numpy().flatten()
prelab = prelab.numpy().flatten()
for ii,lab in enumerate(prelab):
    print("index:",lab," label: ",labels_orig[str(lab)]," ||",prob[ii])
im_pre[:, prelab[0]].backward(retain_graph=None, create_graph=False)#不知道为什么这里会报超时的错
gradients = vggcam.get_activations_gradient()
mean_gradients = torch.mean(gradients,dim=[0,2,3])
activations = vggcam.get_activations(input_im).detach()
for i in range(len(mean_gradients)):
    activations[:, i, :, :] *=mean_gradients[i]
heatmap = torch.mean(activations,dim=1).squeeze()
heatmap = F.relu(heatmap)
heatmap/=torch.max(heatmap)
heatmap = heatmap.numpy()
plt.matshow(heatmap)
img = cv2.imread("D:/train_image/n0/crocodile.jpg")#这里的路径不能有中文，否则会报错
heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
heatmap = np.uint8(255*heatmap)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
Grad_cam_img = heatmap * 0.4+img
Grad_cam_img = Grad_cam_img/Grad_cam_img.max()
b,g,r = cv2.split(Grad_cam_img)
Grad_cam_img = cv2.merge([r,g,b])
plt.figure()
plt.imshow(Grad_cam_img)
plt.show()
