# jonily_zhang
# LSTM进行中文新闻分类
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname="/Library/Fonts/华文细黑.ttf")#没有下载这个字体包后面会报错
import re
import string 
import copy
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import jieba
from torchtext import data
from torchtext.vocab import Vectors

from torchtext.legacy.data import Field,TabularDataset,Iterator,BucketIterator
train_df = pd.read_csv("D:/LSTM进行中文新闻分类/cnews.train.txt",sep="\t",header=None,names=["label","text"])
val_df = pd.read_csv("D:/LSTM进行中文新闻分类/cnews.val.txt",sep="\t",header=None,names=["label","text"])
test_df= pd.read_csv("D:/LSTM进行中文新闻分类/cnews.test.txt",sep="\t",header=None,names=["label","text"])
stop_words = pd.read_csv("D:/LSTM进行中文新闻分类/cn_stopwords.txt",header=None,names=["text"])#读取一个中文停用词数据集
# 这里的中文停用词数据集需要自己另外下载

def chinese_pre(text_data):#定义函数对中文文本数据进行预处理，去除一些不需要的字符、分词、停用词等操作
    text_data = text_data.lower()#字母转小写，去除数字
    text_data = re.sub("\d+","",text_data)
    text_data = list(jieba.cut(text_data,cut_all=False))#分词，使用精确模式
    text_data = [word.strip() for word in text_data if word not in stop_words.text.values]#去停用词和多余空格
    text_data = " ".join(text_data)#处理后的词语使用空格连接为字符串
    return text_data
  
# 对数据进行分词
jieba.setLogLevel(jieba.logging.INFO)
train_df["cutword"] = train_df.text.apply(chinese_pre)
#.apply()函数，读取前面的数据表，默认将每行的数据放入到函数chinese_pre中，最终将结果转换成一个series数据结构（即一维数组类型）
val_df["cutword"] = val_df.text.apply(chinese_pre)
test_df["cutword"] = test_df.text.apply(chinese_pre)
train_df.cutword.head()
# 不是出问题，是运行时间确实比较长

import torchtext
labelMap = {"体育":0,"娱乐":1,"家居":2,"房产":3,"教育":4,"时尚":5,"时政":6,"游戏":7,"科技":8,"财经":9}
train_df["labelcode"] = train_df["label"].map(labelMap)
#.map()函数，将数据集中的label变量分别对应到0~9，生成新的变量labelcode.
val_df["labelcode"] = val_df["label"].map(labelMap)
test_df["labelcode"] = test_df["label"].map(labelMap)
train_df[["labelcode","cutword"]].to_csv("D:/LSTM进行中文新闻分类/cnews.train2.csv",index=False)
#to_csv写入函数，其中index参数类型为布尔值，默认为True，写入行名称（索引）
#将数据保存为csv格式，方便利用torchtext库对文本数据进行预处理
val_df[["labelcode","cutword"]].to_csv("D:/LSTM进行中文新闻分类/cnews.val2.csv",index=False)
test_df[["labelcode","cutword"]].to_csv("D:/LSTM进行中文新闻分类/cnews.test2.csv",index=False)
# 使用torchtext库进行数据准备
mytokenize = lambda x:x.split()
TEXT = torchtext.legacy.data.Field(sequential=True,tokenize=mytokenize,include_lengths=True,use_vocab=True,batch_first=True,fix_length=400)
LABEL = torchtext.legacy.data.Field(sequential=False,use_vocab = False,pad_token = None,unk_token = None)

text_data_fields = [
    ("labelcode",LABEL),
    ("cutword",TEXT)
]
traindata,valdata,testdata = torchtext.legacy.data.TabularDataset.splits(
    path="D:/LSTM进行中文新闻分类",format="csv",
    train="cnews.train2.csv",fields=text_data_fields,
    validation="cnews.val2.csv",test="cnews.test2.csv",skip_header=True)
len(traindata),len(valdata),len(testdata)

# from cmd_color_printers import *
TEXT.build_vocab(traindata,max_size=20000,vectors=None)
LABEL.build_vocab(traindata)
word_fre = TEXT.vocab.freqs.most_common(n=50)
word_fre = pd.DataFrame(data=word_fre,columns=["word","fre"])
word_fre.plot(x="word",y="fre",kind="bar",legend=False,figsize=(12,7))
# plt.xticks(rotation=90,fontproperties=fonts,size=10)
plt.xticks(rotation=90,fontproperties="STXingkai",size=10)
# 如果这里没有下载对应的字体包，那么得自己再去找一个字体函数，否则后面会输出空格报错，而且这里不能空
plt.show()

BATCH_SIZE = 64
train_iter = torchtext.legacy.data.BucketIterator(traindata,batch_size=BATCH_SIZE)
val_iter = torchtext.legacy.data.BucketIterator(valdata,batch_size = BATCH_SIZE)
test_iter = torchtext.legacy.data.BucketIterator(testdata,batch_size=BATCH_SIZE)

class LSTMNet(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,layer_dim,output_dim):
        super(LSTMNet,self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,layer_dim,batch_first=True)
        self.fc1 = nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        embeds = self.embedding(x)
        r_out,(h_n,h_c) = self.lstm(embeds,None)
        out = self.fc1(r_out[:,-1,:])
        return out
    
import pickle as pkl
vocab_size = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 128
layer_dim = 1
output_dim = 10
lstmmodel = LSTMNet(vocab_size,embedding_dim,hidden_dim,layer_dim,output_dim)
lstmmodel
torch.save(lstmmodel,"D:/LSTM进行中文新闻分类/lstmmodel.pkl")#这里要记得保存模型

def train_model2(model,traindataloader,valdataloader,criterion,optimizer,num_epochs=25,):
    train_loss_all=[]
    train_acc_all=[]
    val_loss_all=[]
    val_acc_all=[]
    since=time.time()
    for epoch in range(num_epochs):
        print("-"*10)
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        train_loss=0.0
        train_corrects=0
        train_num=0
        val_loss=0.0
        val_corrects=0
        val_num=0
        model.train()
        for step,batch in enumerate(traindataloader):
            textdata,target = batch.cutword[0],batch.labelcode.view(-1)
            out = model(textdata)
            pre_lab = torch.argmax(out,1)
            loss = criterion(out,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*len(target)
            train_corrects +=torch.sum(pre_lab==target.data)
            train_num+=len(target)
            train_loss_all.append(train_loss/train_num)
            train_acc_all.append(train_corrects.double().item()/train_num)
            print('{} Trian Loss: {:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
            model.eval()
            for step, batch in enumerate(valdataloader):
                textdata,target = batch.cutword[0],batch.labelcode.view(-1)
                out = model(textdata)
                pre_lab = torch.argmax(out,1)
                loss=criterion(out,target)
                val_loss+=loss.item()*len(target)
                val_corrects +=torch.sum(pre_lab == target.data)
                val_num+=len(target)
            val_loss_all.append(val_loss/val_num)
            val_acc_all.append(val_corrects.double().item()/val_num)
            print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(
            epoch,val_loss_all[-1],val_acc_all[-1]))
        data={"epoch":range(num_epochs),
             "train_loss_all":train_loss_all,
             "train_acc_all":train_acc_all,
             "val_loss_all":val_loss_all,
             "val_acc_all":val_acc_all}
        return model,train_process
    
    optimizer = torch.optim.Adam(lstmmodel.parameters(),lr=0.0003)
loss_func = nn.CrossEntropyLoss()
lstmmodel,train_process = train_model2(
    lstmmodel,train_iter,val_iter,loss_func,optimizer,num_epochs=20)
plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.plot(train_precess.epoch,train_process.train_loss_all,"r.-",label="Train Loss")
plt.plot(train_precess.epoch,train_process.val_loss_all,"bs-",label="Val Loss")
plt.legend()
plt.xlabel("Epoch number",size=13)
plt.ylabel("Loss value",size=13)
plt.subplot(1,2,2)
plt.plot(train_process.epoch,train_process.train_acc_all,"r.-",label="Train Acc")
plt.plot(train_process.epoch,train_process.val_acc_all,"bs-",label="Val Acc")
plt.xlabel("Epoch number",size=13)
plt.ylabel("Acc",size=13)
plt.legend()
plt.show()
# 如果卡住了可以选择中断内核再重新运行

lstmmodel.eval()
test_y_all = torch.LongTensor()
pre_lab_all = torch.LongTensor()
for step,batch in enumerate(test_iter):
    textdata,target = batch.cutword[0],batch.labelcode.view(-1)
    out = lstmmodel(textdata)
    pre_lab = torch.argmax(out,1)
    test_y_all = torch.cat((test_y_all,target))
    pre_lab_all = torch.cat((pre_lab_all,pre_lab))
acc = accuracy_score(test_y_all,pre_lab_all)
print("the pre_acc_all in Val:",acc)
class_label = ["体育","娱乐","家居","房产","教育","时尚","时政","游戏","科技","财经"]
conf_mat=confusion_matrix(test_y_all,pre_lab_all)
df_cm = pd.DataFrame(conf_mat,index = class_label,columns=class_label)
heatmap = sns.heatmap(df_cm,annot = True,fmt="d",cmap="YlGnBu")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),rotation=0,ha='right',fontproperties="STXingkai")
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),rotation=45,ha='right',fontproperties="STXingkai")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.manifold import TSNE
lstmmodel = torch.load("D:/LSTM进行中文新闻分类/lstmmodel.pkl")
word2vec = lstmmodel.embedding.weight
words = TEXT.vocab.itos
tsne = TSNE(n_components=2,random_state=123)
word2vec_tsne=tsne.fit_transform(word2vec.data.numpy())
plt.figure(figsize=(10,8))
plt.scatter(word2vec_tsne[:,0],word2vec_tsne[:,1],s=4)
plt.title("所有词向量的分布情况：",fontproperties="STXinwei",size=15)
plt.show()

vis_word=["中国","市场","公司","美国","记者","学生","游戏","北京","投资","电影","银行",
          "工作","留学","大学","经济","产品","设计","方面","玩家","学校","学习","房价","专家","楼市"
          ]
vis_word_index = [words.index(ii) for ii in vis_word]
plt.figure(figsize=(10,8))
for ii,index in enumerate(vis_word_index):
    plt.scatter(word2vec_tsne[index,0],word2vec_tsne[index,1])
    plt.text(word2vec_tsne[index,0],word2vec_tsne[index,1],vis_word[ii],
            fontproperties="STXinwei")
plt.title("词向量的分布情况",fontproperties="STXinwei",size=15)
plt.show()
