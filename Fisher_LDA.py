#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from numpy.linalg import pinv
import matplotlib.lines as mlines
plt.rcParams["axes.grid"] = False
# plt.style.use('ggplot')


# In[ ]:


##-----Fishers Linear Discriminant - Intuitio


# In[3]:


Ax = np.random.uniform(low=-1.8,high=4.0,size=[50,1])
Ay = np.random.uniform(low=2,high=4.1,size=[50,1])
classA = np.column_stack((Ax,Ay))

Bx = np.random.uniform(low=1.5,high=7.0,size=[50,1])
By = np.random.uniform(low=0.1,high=2.1,size=[50,1])
classB = np.column_stack((Bx,By))

labelA = np.zeros((classA.shape[0]),dtype=np.int8)
labelB = np.ones((classB.shape[0]),dtype=np.int8)
targets = np.append(labelA,labelB,axis=0)

data = np.append(classA,classB,axis=0)
data_dict = {0: classA, 1:classB}

print(data.shape)
print(targets.shape)


# In[4]:


Ax = np.random.normal(1.2, 1.0, 50)
Ay = np.random.normal(2.9, 0.3, 50)
classA = np.column_stack((Ax,Ay))

Bx = np.random.normal(3.8, 1.0, 50)
By = np.random.normal(1.6, 0.3, 50)
classB = np.column_stack((Bx,By))

labelA = np.zeros((classA.shape[0]),dtype=np.int8)
labelB = np.ones((classB.shape[0]),dtype=np.int8)
targets = np.append(labelA,labelB,axis=0)

data = np.append(classA,classB,axis=0)
data_dict = {0: classA, 1:classB}

print(data.shape)
print(targets.shape)


# In[5]:


plt.figure(figsize=(10,8))
colors=['red','blue']
for point,pred in zip(data,targets):
  plt.scatter(point[0],point[1],color=colors[pred])
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()


# In[6]:


# compute the mean vector for each of the 2 classes
# compute the mean accros the Bash size dimension
m1 = np.mean(data_dict[0],axis=0)
m2 = np.mean(data_dict[1],axis=0)
print(m1,m2)


# In[7]:


fig, ax = plt.subplots(figsize=(10,8))

colors=['red','blue']
for point,pred in zip(data,targets):
  ax.scatter(point[0],point[1],color=colors[pred], alpha=0.4)

line = mlines.Line2D([m1[0],m2[0]], [m1[1],m2[1]], color='green')
ax.add_line(line)

# plot the mean point of each class
ax.scatter(m1[0],m1[1],color='magenta',s=100,marker="X")
ax.scatter(m2[0],m2[1],color='orange',s=100,marker="X")

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()


# In[8]:


# find the line that joins to 2 class means
print(m2,m1)
# To find the line that join the 2 class means, we can use the slope-intercept form to find the equation of the line from 2 points. 
sub = np.subtract(m2,m1)
print(sub)
m = sub[1]/sub[0]
print("Slope:,",m)
b = -(m * m1[0]) + m1[1]
print("intercept:",b)


# In[9]:


W =np.subtract(m2,m1)
print("Weights:",W)


# In[10]:


fig, ax = plt.subplots(figsize=(10,8))

colors=['red','blue']
for point,pred in zip(data,targets):
  ax.scatter(point[0],point[1],color=colors[pred], alpha=0.15)
  proj = np.dot(point,W)/np.dot(W,W) * W
  #print(proj.shape)
  ax.scatter(proj[0],proj[1],color=colors[pred])
  
  #y = np.dot(point,W)
  #ax.scatter(y,y,color=colors[pred])
  
# plot the mean point of each class
ax.scatter(m1[0],m1[1],color='magenta',s=100,marker="X")
ax.scatter(m2[0],m2[1],color='orange',s=100,marker="X")

line = mlines.Line2D([m1[0],m2[0]], [m1[1],m2[1]], color='green')
#line = mlines.Line2D([0,W[0]], [0,W[1]], color='red')

#transform = ax.transAxes
#line.set_transform(transform)
ax.add_line(line)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()


# In[11]:


fig, ax = plt.subplots(figsize=(10,7))

colors=['red','blue']
for point,pred in zip(data,targets):
  ax.scatter(point[0],point[1],color=colors[pred],alpha=0.15)
  proj = np.dot(point,W)/np.dot(W,W) * W
  y = np.dot(point,W)

  #print(proj.shape)
  ax.scatter(proj[0],proj[1],color=colors[pred])
  #ax.scatter(y,y,color=colors[pred])

# plot the mean point of each class
ax.scatter(m1[0],m1[1],color='magenta',s=100,marker="X")
ax.scatter(m2[0],m2[1],color='orange',s=100,marker="X")

#line = mlines.Line2D([0,W[0]], [0,W[1]], color='red')
#transform = ax.transAxes
#line.set_transform(transform)
#ax.add_line(line)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()


# In[ ]:




