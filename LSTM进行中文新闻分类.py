# jonily_zhang
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
test
