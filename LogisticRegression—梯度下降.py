#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""导包"""
import random
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def initTheta(Theta):
    """初始化Theta"""
    for i in range(len(Theta)):
        Theta[i]=random.uniform(-10,10)


# In[3]:


def sigmoid(z):
    """sigmoid函数"""
    return 1.0/(1.0+np.exp(-z))


# In[4]:


def costFunction(x,y,Theta,regularization,index):
    """代价函数"""
    sums=0
    for i in range(len(x)):
        r=0
        for j in range(len(Theta)):
            r+=Theta[j]*x[i][j]
        sums+=(sigmoid(r)-y[i])*x[i][index]
    return 1.0/len(x)*(sums+regularization*Theta[index])


# In[5]:


def train(x,y,learnRate,iterations,regularization):
    """训练数据"""
    Theta=[0 for i in range(len(x[0]))]
    initTheta(Theta)
    t=0
    while t<iterations:
        for i in range(len(Theta)):
            Theta[i]-=learnRate*costFunction(x,y,Theta,regularization,i)
        t+=1
    return Theta


# In[6]:


def test(Theta,x):
    """测试数据"""
    Result=[0 for i in range(len(x))]
    for i in range(len(x)):
        sums=0
        for j in range(len(Theta)):
            sums+=Theta[j]*x[i][j]
        temp=sigmoid(sums)
        if temp<0.5:
            Result[i]=0
        else:
            Result[i]=1
    return Result


# In[7]:


def deviationFunction(y,result):
    """误差函数"""
    Deviation=[0 for i in range(len(y))]
    for i in range(len(y)):
        Deviation[i]=abs(y[i]-result[i])
    return Deviation


# In[8]:


def drawDeviation(y):
    """画出误差图像"""
    x=np.linspace(0,len(Deviation),len(Deviation))
    plt.plot(x,y,color='black',  # 线条颜色
             linewidth = 1.5,  # 线条宽度
             linestyle='-'  # 线条样式
        )
    plt.show()


# In[9]:


#超参数
learnRate=0.01
iterations=10000
regularization=10

#数据集
trainX=[[1,5,3],[1,-2,2],[1,3,7],[1,2,5],[1,4,-4],[1,5,9],[1,18,3],[1,0,-5],[1,2,1],[1,-6,0],[1,-10,10],[1,21,1],[1,6,1],[1,-3,5],[1,0,6]]
trainY=[1,1,0,1,1,0,0,1,1,1,0,0,0,1,1]
testX=[[1,7,1],[1,3,4],[1,8,-4],[1,4,0],[1,0,0],[1,6,2],[1,9,4]]
testY=[0,1,0,1,1,0,0]


# In[10]:


#测试
Theta=train(trainX,trainY,learnRate,iterations,regularization)
Result=test(Theta,testX)
Deviation=deviationFunction(testY,Result)
print("权重值：",Theta)
print("标准值：",testY)
print("测试值：",Result)
print("误差值：",Deviation)
print("\n误差图：")
drawDeviation(Deviation)

