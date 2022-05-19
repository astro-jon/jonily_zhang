# jonily_zhang
# 导入需要的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import seaborn as sns
from wordcloud import WordCloud
import time
import copy
import torch
import torchtext
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchtext import data
# from torchtext.legacy.data import Field
from torchtext.vocab import Vectors,GloVe

# 由于教材原数据找不到，以下操作是对数据进行预处理，整理成教材的模型
path="D:/IMDB电影评论情感分析/test.csv"
def save(filename, contents):
    fh = open(filename, 'w', encoding='utf-8')#open有该文件则打开，没有该文件就创建
    fh.write(contents)
    fh.close()
flag=0
neg=0
pos=0
with open(path, 'r', encoding='utf-8-sig') as test:
    for line in test:
        lines = line[:-5]#删除字符串后五位
        if flag==0:
            flag+=1
        else:
            sentiment=line[-4]+line[-3]+line[-2]
            if sentiment=='neg':
                filename_neg="D:/IMDB电影评论情感分析/neg/_"+str(neg)+'.txt'
                save(filename_neg,lines)
                neg=int(neg)+1
            else:
                filename_pos="D:/IMDB电影评论情感分析/pos/_"+str(pos)+'.txt'
                save(filename_pos,lines)
                pos=int(pos)+1
path="D:/IMDB电影评论情感分析/train.csv"
def save(filename, contents):
    fh = open(filename, 'w', encoding='utf-8')#open有该文件则打开，没有该文件就创建
    fh.write(contents)
    fh.close()
flag=0
neg=0
pos=0
with open(path, 'r', encoding='utf-8-sig') as test:
    for line in test:
        lines = line[:-5]#删除字符串后五位
        if flag==0:
            flag+=1
        else:
            sentiment=line[-4]+line[-3]+line[-2]
            if sentiment=='neg':
                filename_neg="D:/IMDB电影评论情感分析/train/neg/_"+str(neg)+'.txt'
                print(filename_neg)
                save(filename_neg,lines)
                neg=int(neg)+1
            else:
                filename_pos="D:/IMDB电影评论情感分析/train/pos/_"+str(pos)+'.txt'
                print(filename_pos)
                save(filename_pos,lines)
                pos=int(pos)+1
                
# 定义读取训练数据和测试数据的函数               
def load_text_data(path):
#   获取文件夹最后一个字段
    text_data=[]
    label=[]
    for dset in ["pos","neg"]:
        path_dset = os.path.join(path,dset)
        path_list = os.listdir(path_dset)
#         读取文件夹下的pos或者neg文件
        for fname in path_list:
            if fname.endswith(".txt"):
                filename = os.path.join(path_dset,fname)
                with open(filename,'rb') as f:#这里open()函数里面的打开方式要改成"rb",因为原先默认为"b"二进制打开
                    text_data.append(f.read())
            if dset =="pos":
                label.append(1)
            else:
                label.append(0)
#     输出读取文本和对应的标签
    return np.array(text_data),np.array(label)
# 读取训练集和测试集
train_path = "D:/IMDB电影评论情感分析/train"
train_text,train_label=load_text_data(train_path)
test_path = "D:/IMDB电影评论情感分析/test"
test_text,test_label = load_text_data(test_path)
print(len(train_text),len(train_label))
print(len(test_text),len(test_label)) 
# html=html.decode('utf-8')#python3
# 文本数据预处理
def text_preprocess(text_data):
    text_pre=[]
    for text1 in text_data:
        text1 = str(text1)
        text1 = re.sub("<br /><br />"," ",text1)
        text1 = text1.lower()#转小写
        text1 = re.sub("\d+","",text1)
        text1 = text1.translate(str.maketrans("","",string.punctuation.replace("'","")))
        text1 = text1.strip()
        text1 = text1[2:-1]
        text_pre.append(text1)
    return np.array(text_pre)
train_text_pre = text_preprocess(train_text)
test_text_pre = text_preprocess(test_text)

# 文本符号化处理，去除停用词
# 停用词要导入这个函数库
import nltk
# nltk.download('stopwords')
def stop_stem_word(datalist,stop_words):
    datalist_pre=[]
    for text in datalist:
        text_words = nltk.word_tokenize(text)
#         去除停用词
        text_words = [word for word in text_words if not word in stop_words]
#         删除带有“’”的词语
        text_words = [word for word in text_words if len(re.findall("'",word))==0]
        datalist_pre.append(text_words)
    return np.array(datalist_pre)
# stop_words = nltk.corpus.stopwords("english")
stop_words = stopwords.words("english")
stop_words = set(stop_words)
train_text_pre2 = stop_stem_word(train_text_pre,stop_words)
test_text_pre2 = stop_stem_word(test_text_pre,stop_words)
print(train_text_pre[10000])
print("="*10)
print(train_text_pre2[10000])

# 将处理好的文本保存到csv文件中
texts=[" ".join(words) for words in train_text_pre2]
traindatasave = pd.DataFrame({"text":texts,"label":train_label})
tests = [" ".join(words) for words in test_text_pre2]
testdatasave = pd.DataFrame({"text":texts,"label":test_label})
traindatasave.to_csv("D:/IMDB电影评论情感分析/imdb_train.csv",index = False)
testdatasave.to_csv("D:/IMDB电影评论情感分析/imdb_test.csv",index = False)

# 将预处理好的文本数据转换为数据表
traindata = pd.DataFrame({"train_text":train_text,"train_word":train_text_pre2,"train_label":train_label})
# 计算每一个影评使用词的数量
train_word_num = [len(text) for text in train_text_pre2]
traindata["train_word_num"] = train_word_num
# 可视化影评词长度的分布
plt.figure(figsize=(8,5))
_ = plt.hist(train_word_num,bins=100,color='pink')#第一个为输入的数据，第二个为可视化的箱子数（及柱数），第三个为颜色
plt.xlabel("word number")
plt.ylabel("Freq")
plt.show()

# 使用词云可视化两种情感的词频差异
plt.figure(figsize=(16,10))
for ii in np.unique(train_label):
    text = np.array(traindata.train_word[traindata.train_label == ii])
    text = " ".join (np.concatenate(text))
    plt.subplot(1,2,ii+1)
    wordcod = WordCloud(margin=5,width=1800,height = 1000,max_words = 500,min_font_size=5,background_color = 'black',max_font_size=250)
    wordcod.generate_from_text(text)
    plt.imshow(wordcod)
    plt.axis("off")
    if ii ==1:
        plt.title("Positive")
    else:
        plt.title("Negative")
    plt.subplots_adjust(wspace=0.05)
plt.show()

# 使用torchtext库进行数据准备，定义文件中对文本和标签所要做的操作
# 定义文本切分方法，因为前面已经做过处理，所以直接使用空格切分即可
mytokenize = lambda x:x.split()
from torchtext.legacy.data import Field,TabularDataset,Iterator,BucketIterator
#Field封装在.legacy中
#对于torch1.0及以上版本，.legacy()已经给舍弃，需要对torch进行降级才能使用
# 关于torch下载直接去官网，找寻对应的版本在anaconda页面下下载即可，且不会影响到其他函数包的使用
TEXT = torchtext.legacy.data.Field(sequential=True,tokenize=mytokenize,include_lengths=True,use_vocab=True,batch_first=True,fix_length=200)
LABEL = torchtext.legacy.data.Field(sequential=False,use_vocab = False,pad_token= None,unk_token=None)
train_test_fields=[
    ("text",TEXT),
    ("label",LABEL)
]#由于自己预处理的数据排序问题，text要放在前面，不然在后面函数传值时text与label的顺序会弄反
traindata,testdata = torchtext.legacy.data.TabularDataset.splits(
    path="D:/IMDB电影评论情感分析",format="csv",
    train="imdb_train.csv",
#     fields=train_test_fields,
    fields=list(train_test_fields),#按照书籍源码要求，这里数据要处理成列表形式
    test = "imdb_test.csv",skip_header = True)
len(traindata),len(testdata)
ex0 = traindata.examples[0]
# print(ex0.label)
# print(ex0.text)
# 将训练集切分为训练集与测试集
train_data,val_data = traindata.split(split_ratio=0.7)
# len(train_data),len(val_data)
vec = Vectors("glove.6B.100d.txt","D:/预训练好的词向量")
TEXT.build_vocab(train_data,max_size=20000,vectors = vec)
LABEL.build_vocab(train_data)
print(TEXT.vocab.freqs.most_common(n=10))
print("词典的词数：",len(TEXT.vocab.itos))
print("the front 10 words:\n",TEXT.vocab.itos[0:10])
print("classification label situation :",LABEL.vocab.freqs)
BATCH_SIZE=32
train_iter = torchtext.legacy.data.BucketIterator(train_data,batch_size=BATCH_SIZE)
val_iter = torchtext.legacy.data.BucketIterator(val_data,batch_size = BATCH_SIZE)
test_iter = torchtext.legacy.data.BucketIterator(testdata,batch_size = BATCH_SIZE)
for step,batch in enumerate(train_iter):
    if step>0:
        break
# 针对一个batch的数据，可以使用batch.label获得数据的类别标签        
# print("the data's classification label:\n",batch.label)
# print("the data size:",batch.text[0].shape)
# print("the data sample number:",len(batch.text[1]))
class CNN_Text(nn.Module):
    def __init__(self,vocab_size,embedding_dim,n_filters,filter_sizes,output_dim,
                dropout,pad_idx):
        super().__init__()
#         对文本进行词嵌入操作
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_idx)
#         卷积操作
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels = 1,out_channels=n_filters,kernel_size=(fs,embedding_dim)) for fs in filter_sizes
        ])
#         全连接层和Dropout层
        self.fc = nn.Linear(len(filter_sizes)*n_filters,output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled,dim=1))
        return self.fc(cat)
      
INPUT_DIM = len(TEXT.vocab)#词典的数量
EMBEDDING_DIM=100 #词向量的维度
N_FILTERS = 100 #每个卷积核的个数
FILTER_SIZES = [3,4,5] #卷积核的高度
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] #填充词的索引
model = CNN_Text(INPUT_DIM,EMBEDDING_DIM,N_FILTERS,FILTER_SIZES,OUTPUT_DIM,DROPOUT,PAD_IDX)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
def train_epoch(model,iterator,optimizer,criterion):
    epoch_loss = 0;
    epoch_acc = 0
    train_corrects = 0;train_num = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        pre = model(batch.text[0]).squeeze(1)
        loss  = criterion(pre,batch.label.type(torch.FloatTensor))
        pre_lab = torch.round(torch.sigmoid(pre))
        train_corrects += torch.sum(pre_lab.long() == batch.label)
        train_num+=len(batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    epoch_loss=epoch_loss/train_num
    epoch_acc=train_corrects.double().item() / train_num
    return epoch_loss,epoch_acc
def evaluate(model,iterator,criterion):
    epoch_loss=0;epoch_acc=0
    train_corrects = 0;train_num=0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            pre = model(batch.text[0]).squeeze(1)
            loss = criterion(pre,batch.label.type(torch.FloatTensor))
            pre_lab = torch.round(torch.sigmoid(torch.sigmoid(pre)))
            train_corrects +=torch.sum(pre_lab.long()==batch.label)
            train_num+=len(batch.label)
            epoch_loss+=loss.item()
        epoch_loss=epoch_loss/train_num
        epoch_acc=train_corrects.double().item()/train_num
    return epoch_loss,epoch_acc
EPOCHS = 10
best_val_loss = float("inf")
best_acc = float(0)
for epoch in range(EPOCHS):
    start_time=time.time()
    train_loss,train_acc = train_epoch(model,train_iter,optimizer,criterion)
    val_loss,val_acc = evaluate(model,val_iter,criterion)
    end_time = time.time()
    print("Epoch:",epoch+1,"|","Epoch Time: ",end_time-start_time,"s")
    print("Train Loss:",train_loss,"|","Train acc: ",train_acc)
    print("Val.Loss: ",val_loss,"|","Val.acc: ",val_acc)
    if(val_loss<best_val_loss) &(val_acc>best_acc):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_loss = val_loss
        best_acc = val_acc
model.load_state_dict(best_model_wts)
