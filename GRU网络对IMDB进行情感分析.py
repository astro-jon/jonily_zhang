# jonily_zhang
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchtext import data
from torchtext.vocab import Vectors
#如果遇到“python无法定位程序输入口的话”，删除报错路径的文件

import torchtext
from torchtext.legacy.data import Field,TabularDataset,Iterator,BucketIterator
# 定义文本切分方法，直接使用空格切分即可
mytokenize = lambda x: x.split()#lambda定义一个匿名函数，这里x为输入，x.split()为输出
TEXT = torchtext.legacy.data.Field(sequential=True,tokenize=mytokenize,include_lengths = True,
                                  use_vocab=True,batch_first = True,fix_length=200)
LABEL = torchtext.legacy.data.Field(sequential = False,use_vocab = False,pad_token=None,unk_token = None)
train_test_fields = [
    ("text",TEXT),
    ("label",LABEL)#这里位置别弄反
]
traindata,testdata = torchtext.legacy.data.TabularDataset.splits(
    path="D:/IMDB电影评论情感分析",format = "csv",
    train="imdb_train.csv",fields=train_test_fields,
    test = "imdb_test.csv",skip_header = True)

vec = Vectors("glove.6B.100d.txt","D:/预训练好的词向量")
TEXT.build_vocab(traindata,max_size=20000,vectors = vec)
LABEL.build_vocab(traindata)
BATCH_SIZE = 32
train_iter = torchtext.legacy.data.BucketIterator(traindata,batch_size=BATCH_SIZE)
test_iter = torchtext.legacy.data.BucketIterator(testdata,batch_size=BATCH_SIZE)

class GRUNet(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,layer_dim,output_dim):
        super(GRUNet,self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.gru = nn.GRU(embedding_dim,hidden_dim,layer_dim,batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim,output_dim))
    def forward(self,x):
        embeds = self.embedding(x)
        r_out,h_n = self.gru(embeds,None)
        out = self.fc1(r_out[:,-1,:])
        return out
      
vocab_size=len(TEXT.vocab)
embedding_dim = vec.dim
hidden_dim= 128
layer_dim = 1
output_dim = 2
grumodel = GRUNet(vocab_size,embedding_dim,hidden_dim,layer_dim,output_dim)
grumodel

grumodel.embedding.weight.data.copy_(TEXT.vocab.vectors)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
grumodel.embedding.weight.data[UNK_IDX] = torch.zeros(vec.dim)
grumodel.embedding.weight.data[PAD_IDX] = torch.zeros(vec.dim)

def train_model(model,traindataloader,testdataloader,criterion,optimizer,num_epochs = 25):
    train_loss_all=[]
    train_acc_all=[]
    test_loss_all=[]
    test_acc_all=[]
    learn_rate=[]
    since = time.time()
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 5,gamma=0.1)
    for epoch in range(num_epochs):
        learn_rate.append(scheduler.get_lr()[0])
        print('-'*10)
        print('Epoch {}/{},Lr:{}'.format(epoch,num_epochs- 1,learn_rate[-1]))
        train_loss=0.0
        train_corrects = 0
        train_num=0
        test_loss = 0.0
        test_corrects = 0
        test_num = 0
        model.train()
        for step,batch in enumerate(traindataloader):
            textdata,target = batch.text[0],batch.label
            out = model(textdata)
            pre_lab = torch.argmax(out,1)
            loss = criterion(out,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*len(target)
            train_corrects += torch.sum(pre_lab == target.data)
            train_num+=len(target)
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        scheduler.step()#更新学习率
        model.eval()
        for step,batch in enumerate(testdataloader):
            textdata,target = batch.text[0],batch.label
            out = model(textdata)
            pre_lab = torch.argmax(out,1)
            loss = criterion(out,target)
            test_loss+=loss.item() * len(target)
            test_corrects += torch.sum(pre_lab == target.data)
            test_num +=len(target)
        test_loss_all.append(test_loss/test_num)
        test_acc_all.append(test_corrects.double().item()/test_num)
        print('{} Test Loss: {:.4f} Test Acc: {:.4f}'.format(epoch,test_loss_all[-1],test_acc_all[-1]))
    train_process = pd.DataFrame(
        data={"epoch":range(num_epochs),
             "train_loss_all":train_loss_all,
             "train_acc_all":train_acc_all,
             "test_loss_all":test_loss_all,
             "test_acc_all":test_acc_all,
             "learn_rate":learn_rate})
    return model,train_process
  
optimizer = optim.RMSprop(grumodel.parameters(),lr=0.003)
loss_func = nn.CrossEntropyLoss()
grumodel,train_process = train_model(
    grumodel,train_iter,test_iter,loss_func ,optimizer,num_epochs = 10)

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.plot(train_process.epoch,train_process.train_loss_all,
        "r.-",label="Train loss")
plt.plot(train_process.epoch,train_process.test_loss_all,
        "bs-",label="Test loss")
plt.legend()
plt.xlabel("Epoch number",size = 13)
plt.ylabel("Loss value",size=13)
plt.subplot(1,2,2)
plt.plot(train_process.epoch,train_process.train_acc_all,
        "r.-",label = "Test acc")
plt.plot(train_process.epoch,train_process.test_acc_all,
        "bs-",label="Test acc")
plt.xlabel("Epoch number",size=13)
plt.ylabel('Acc',size=13)
plt.legend()
plt.show()

grumodel.eval()
test_y_all = torch.LongTensor()
pre_lab_all=torch.LongTensor()
for step,batch in enumerate(test_iter):
    textdata,target = batch.text[0],batch.label.view(-1)
    out = grumodel(textdata)
    pre_lab = torch.argmax(out,1)
    test_y_all = torch.cat((test_y_all,target))
    pre_lab_all = torch.cat((pre_lab_all,pre_lab))
acc = accuracy_score(test_y_all,pre_lab_all)
print("the accuracy of the test:",acc)
