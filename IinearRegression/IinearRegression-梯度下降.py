#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""导包"""
import random
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def initTheta(Theta):
    """初始化Theta值"""
    for i in range(len(Theta)):
        Theta[i]=random.uniform(-10,10)


# In[3]:


def costFunction(x,y,Theta,index,regularization):
    """代价函数"""
    sums=0
    for i in range(len(x)):
        r=0
        for j in range(len(Theta)):
            r+=Theta[j]*x[i][j]
        sums+=(r-y[i])*x[i][index]
    return 1/len(x)*(sums+regularization*Theta[index])


# In[4]:


def train(x,y,learnRate,iterations,regularization):
    """训练数据"""
    Theta=[0 for i in range(len(x[0]))]
    initTheta(Theta)
    t=0
    while t<iterations:
        for i in range(len(Theta)):
            Theta[i]-=learnRate*costFunction(x,y,Theta,i,regularization)
        t+=1
    return Theta


# In[5]:


def test(x,y,Theta):
    """测试数据"""
    result=[0 for i in range(len(y))]
    for i in range(len(x)):
        sums=0
        for j in range(len(Theta)):
            sums+=Theta[j]*x[i][j]
        result[i]=sums
    return result


# In[6]:


def deviationFunction(y,result):
    """误差函数"""
    Deviation=[0 for i in range(len(y))]
    for i in range(len(y)):
        Deviation[i]=abs(y[i]-result[i])
    return Deviation


# In[7]:


def drawDeviation(y):
    """画出误差图像"""
    x=np.linspace(0,len(Deviation),len(Deviation))
    plt.plot(x,y,color='black',  # 线条颜色
             linewidth = 1.5,  # 线条宽度
             linestyle='-'  # 线条样式
        )
    plt.show()


# In[8]:


#参数
learnRate=0.0001
iterations=10000
regularization=100

#数据集
trainX=[[1,1,1],[1,2,2],[1,3,3],[1,4,4],[1,5,5],[1,29,29],[1,100,100],[1,-3,-3],[1,63,63],[1,94,94]]
trainY=[2,4,6,8,10,29,100,-3,63,94]
testX=[[1,10,10],[1,30,30],[1,-1,-1],[1,8,8],[1,91,91],[1,-20,-20]]
testY=[10,30,-1,8,91,-20]


# In[9]:


#测试
Theta=train(trainX,trainY,learnRate,iterations,regularization)
result=test(testX,testY,Theta)
Deviation=deviationFunction(testY,result)
print("权重值：",Theta)
print("标准值：",testY)
print("测试值：",result)
print("误差值：",Deviation)
print("\n误差图：")
drawDeviation(Deviation)

