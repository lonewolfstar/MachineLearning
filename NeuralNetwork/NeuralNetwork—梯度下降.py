#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""导包"""
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def initParameters(layers_dim):
    """初始化权重"""
    L=len(layers_dim)
    parameters={}
    for i in range(1,L):
        parameters["w"+str(i)]=np.random.random((layers_dim[i],layers_dim[i-1]))
        parameters["b"+str(i)]=np.random.random((layers_dim[i],1))
    return parameters


# In[3]:


def sigmoid(z):
    """sigmoid函数"""
    return 1.0/(1.0+np.exp(-z))


# In[4]:


def sigmoidPrime(z):
    """sigmoid函数的导数"""
    return sigmoid(z)*(1-sigmoid(z))


# In[5]:


def forward(x,parameters):
    """前向传播"""
    a=[]
    z=[]
    caches={}
    a.append(x)
    z.append(x)
    layers=len(parameters)//2
    
    for i in range(1,layers):
        z_temp=parameters["w"+str(i)].dot(a[i-1])+parameters["b"+str(i)]
        z.append(z_temp)
        a.append(sigmoid(z_temp))
        
    z_temp=parameters["w"+str(layers)].dot(a[layers-1])+parameters["b"+str(layers)]
    z.append(z_temp)
    a.append(z_temp)
    
    caches["z"]=z
    caches["a"]=a
    
    return caches,a[layers]


# In[6]:


def backward(parameters,caches,al,y):
    """后向传播"""
    grades={}
    m=y.shape[1]
    layers=len(parameters)//2
    
    grades["dz"+str(layers)]=al-y
    grades["dw"+str(layers)]=grades["dz"+str(layers)].dot(caches["a"][layers-1].T)/m
    grades["db"+str(layers)]=np.sum(grades["dz"+str(layers)],axis=1,keepdims=True)/m
    
    t=layers-1
    while t>0:
        grades["dz"+str(t)]=parameters["w"+str(t+1)].T.dot(grades["dz"+str(t+1)])*sigmoidPrime(caches["z"][t])
        grades["dw"+str(t)]=grades["dz"+str(t)].dot(caches["a"][t-1].T)/m
        grades["db"+str(t)]=np.sum(grades["dz"+str(t)],axis=1,keepdims=True)/m
        t-=1
    return grades


# In[7]:


def updateParameters(parameters,grades,learnRate):
    """更新权重"""
    layers=len(parameters)//2
    for i in range(1,layers+1):
        parameters["w"+str(i)]-=learnRate*grades["dw"+str(i)]
        parameters["b"+str(i)]-=learnRate*grades["db"+str(i)]
    return parameters


# In[8]:


def computeLoss(al,y):
    """计算误差"""
    return np.mean(np.square(al-y))


# In[9]:


def loadData():
    """加载数据集"""
#     x=np.arange(0.0,1,0.01)
#     y=20* np.sin(2*np.pi*x)
    x=np.arange(0.0,10,0.1)
    y=pow(x,2)
    plt.scatter(x,y)
    return x,y


# In[10]:


#测试
learnRate=0.01

x,y=loadData()
x=x.reshape(1,100)
y=y.reshape(1,100)
plt.scatter(x,y)

parameters=initParameters([1,25,1])
        
for i in range(10000):
    caches,al=forward(x,parameters)
    grades=backward(parameters,caches,al,y)
    parameters=updateParameters(parameters,grades,learnRate)
    error=computeLoss(al,y)
print(error)
plt.scatter(x,al)
plt.show()


# In[ ]:




