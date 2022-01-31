## P1909 [NOIP2016 普及组] 买铅笔(python解法)
### 题目
![图片](https://user-images.githubusercontent.com/91021948/151737541-969742fb-f74a-463e-9cd5-69d831fe6ddd.png)
- 关于这道题可以使用math中自带库函数ceil：向上取整
### python源代码如下：
## from math import ceil
## a=int(input())
## b,b1=list(map(int,input().split(" ")))
## c,c1=list(map(int,input().split(" ")))
## d,d1=list(map(int,input().split(" ")))
## bn=ceil(a/b)*b1
## cn=ceil(a/c)*c1
## dn=ceil(a/d)*d1
## print(min([bn,cn,dn]))
