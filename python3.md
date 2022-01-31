## P1055 [NOIP2008 普及组] ISBN 号码(python解法)
### 题目描述：
![图片](https://user-images.githubusercontent.com/91021948/151743325-19d4f779-c1f1-41a8-b928-c9a5459aa7eb.png)
### 解题思路：
  这道题首先是要接收一串由数字和"-"组成的字符，可以通过split()或者循环展示来提取中间的数字，再通过题目的公式进行if分类讨论即可

### 源码如下：
![图片](https://user-images.githubusercontent.com/91021948/151744069-c1ff7216-bd5a-4518-81b7-eae82fd33ed2.png)

#### 提醒：
如果选择将每一个数字挑出来再进行计算，很可能会出现超时的问题（本人就是卡在这超时卡了好久），故建议可以选择sum这样累加来做。
