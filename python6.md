## P5729 【深基5.例7】工艺品制作(python解法)
### 题目解析：
![图片](https://user-images.githubusercontent.com/91021948/152493740-53a2776a-7cdd-4677-9faf-67c7bbffd79d.png)
一个立方体从中切割然后计算蒸发后的小方块体积，我首先想到的是创建一个三维数组
#### python中三维数组的创建方式
- import numpy as np#首先调用numpy库
- data_array=np.zeros((4,4,4),dtype=np.int)#这里创建的是一个四行四列乘四的起始值为0的三维实数数组
#### 然后我的想法是两次循环嵌套，第一次是接收小方块位置的输入以及计算蒸发的小方块数量
第二次是将原来的大立方体中所有小立方块检验做标志，将被蒸发的小立方块依次累加，最后再减去这个和即可得答案
### 源码如下：
(这是第一次循环)
![图片](https://user-images.githubusercontent.com/91021948/152495283-953c4f8c-f62b-43c2-899d-5e0052adc295.png)
(这是第二次循环)
![图片](https://user-images.githubusercontent.com/91021948/152495395-f1bc2916-b4ce-4cdf-88a7-e2697a4939ee.png)
#### 希望能对你有用
