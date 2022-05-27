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
# Field参数说明
# squential：数据是否为序列数据，默认为Ture。如果为False，则不能使用分词。
# use_vocab：是否使用词典，默认为True。如果为False，那么输入的数据类型必须是数值类型(即使用vocab转换后的)。
# init_token：文本的其实字符，默认为None。
# eos_token：文本的结束字符，默认为None。
# fix_length：所有样本的长度，不够则使用pad_token补全。默认为None，表示灵活长度。
# tensor_type：把数据转换成的tensor类型 默认值为torch.LongTensor。
# preprocessing：预处理pipeline， 用于分词之后、数值化之前，默认值为None。
# postprocessing：后处理pipeline，用于数值化之后、转换为tensor之前，默认为None。
# lower：是否把数据转换为小写，默认为False；
# tokenize：分词函数，默认为str.split。（tokenize必须是个函数，且返回的是一个列表）
# include_lengths：是否返回一个已经补全的最小batch的元组和和一个包含每条数据长度的列表，默认值为False。
# batch_first：batch作为第一个维度；
# pad_token：用于补全的字符，默认为<pad>。
# unk_token：替换袋外词的字符，默认为<unk>。
# pad_first：是否从句子的开头进行补全，默认为False；
# truncate_first：是否从句子的开头截断句子，默认为False；
# stop_words：停用词；

LABEL = torchtext.legacy.data.Field(sequential = False,use_vocab = False,pad_token=None,unk_token = None)
# 对所要读取的数据集的列进行处理
train_test_fields = [
    ("text",TEXT),
    ("label",LABEL)#这里位置别弄反
]
# 读取数据
traindata,testdata = torchtext.legacy.data.TabularDataset.splits(
    path="D:/IMDB电影评论情感分析",format = "csv",
    train="imdb_train.csv",fields=train_test_fields,
    test = "imdb_test.csv",skip_header = True)
# 如果有列名，设置skip_header为True可以不把列名变成数据处理

# Vectors导入预训练好的词向量文件
vec = Vectors("glove.6B.100d.txt","D:/预训练好的词向量")
# 使用训练集构建单词表，导入预训练好的词嵌入
TEXT.build_vocab(traindata,max_size=20000,vectors = vec)
LABEL.build_vocab(traindata)
# 训练集、验证集和测试集定义为加载器
BATCH_SIZE = 32
train_iter = torchtext.legacy.data.BucketIterator(traindata,batch_size=BATCH_SIZE)
test_iter = torchtext.legacy.data.BucketIterator(testdata,batch_size=BATCH_SIZE)

class GRUNet(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,layer_dim,output_dim):
      """
      vocab_size:词典长度
      embedding_dim：词向量的维度
      hidden_dim：GRU神经元个数
      layer_dim：GRU层数
      output_dim：隐藏层输出的维度(分类的数量）
      """
        super(GRUNet,self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
#         对文本进行词向量处理
#     在 PyTorch 中，针对词向量有一个专门的层 nn.Embedding ，用来实现词与词向量的映射。
#     nn.Embedding 相当于一个词表，形状为 (num_embedings, embedding_dim) ，其中 num_embedding 表示词表的长度，embedding_dim 表示词向量的维度。
#     如果输入张量中不同词的个数超过词表的长度，就会报数组越界异常。
#     如果 Embedding 层的输入形状为 NxM（N为batch_size，M是序列的长度），则输出的形状是 N x M x embedding_dim.
#     注意：输入必须是 LongTensor 类型，可以通过 tensor.long() 方法转成 LongTensor。
#        GRU+全连接层
        self.gru = nn.GRU(embedding_dim,hidden_dim,layer_dim,batch_first=True)
  """
        input_size:输出序列的一维向量的长度
        hidden_size:隐藏层输出特征的长度
        num_size:隐藏层堆叠的高度，用于增加隐层的深度
        bias:是否需要偏置b
        batch_first:用于确定batch_size是否需要放到输入输出数据形状的最前面
        dropout:默认0，若非0，则为dropout率
        bidirectional:是否为双向LSTM，默认为否
  """
#    Embedding 做的事，从数据中自动学习到输入空间的信息表示的映射f，由于上述计算中没有涉及到label ，所以 Embedding 的训练过程是无监督的。
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim,output_dim))
    def forward(self,x):
        embeds = self.embedding(x)
        r_out,h_n = self.gru(embeds,None)# None表示初始的hidden state为0
#         选取最后一个时间点的out输出
        out = self.fc1(r_out[:,-1,:])
        return out

#   初始化网络
vocab_size=len(TEXT.vocab)
embedding_dim = vec.dim #词向量的维度
hidden_dim= 128
layer_dim = 1
output_dim = 2
grumodel = GRUNet(vocab_size,embedding_dim,hidden_dim,layer_dim,output_dim)
grumodel

# 将导入的词向量作为embedding.weight的初始值
grumodel.embedding.weight.data.copy_(TEXT.vocab.vectors)
# 将无法识别的词'<unk>''<pad>'的向量初始化为0
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
grumodel.embedding.weight.data[UNK_IDX] = torch.zeros(vec.dim)
grumodel.embedding.weight.data[PAD_IDX] = torch.zeros(vec.dim)

def train_model(model,traindataloader,testdataloader,criterion,optimizer,num_epochs = 25):
  """
  criterion:损失函数
  optimizer:优化方法
  num_epochs:训练的轮次
  """
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
