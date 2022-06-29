# jonily_zhang
# 半监督图卷积神经网络实战
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
# 首先这里的geometric模块导入一定要根据自己的torch版本下载
# 下载了geometric之后的这些模块的下载也是要根据自己的版本来下载
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.semi_supervised import _label_propagation# 这里导入的时候要看下自己那个包的文件地址下的名字，
# 前面显示还有个_,这里不能少，不然会报错

# from label_propagation import labelPropagation
# from torch_geometric.nn import LabelPropagation
