#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")


# In[3]:


train.head()


# In[4]:


num = np.array(train.iloc[0, 1:]).reshape(28,28)
plt.imshow(num)
plt.show()


# In[5]:


plt.hist(num)
plt.show()


# In[6]:


x = np.array(train)
y = np.array(test)
x_train = x[:, 1:]
y_train = x[:, 0]
x_test = y[:, 1:]
y_test = y[:, 0]


# In[9]:


x_train.shape


# # Feature Enginnering in Images

# In[17]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
x_train = x_train.reshape(-1, 28, 28, 1)


# In[18]:


datagen = ImageDataGenerator(
    rotation_range=15,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
)
datagen.fit(x_train)


# In[19]:


x_train.shape


# In[16]:


x_train = x_train.reshape(-1, 784)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)


# In[ ]:


pred = knn.predict(x_test)


# In[ ]:


pred[100]


# In[ ]:


num = np.array(test.iloc[100, 1:]).reshape(28,28)
plt.imshow(num)
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, pred)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

