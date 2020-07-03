#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection


# In[14]:


dataset = pd.read_csv("D:/Datasets/Covid19Data.csv")


# In[18]:


dataset.shape
df = dataset.dropna()
df.isnull().sum()


# In[21]:


df.shape


# In[28]:


x = np.array(df.drop(['new_cases'],1))
x.shape


# In[30]:


y = np.array(df['new_cases'])
y = y.reshape(len(y),1)
y.shape


# In[31]:


lr = linear_model.LinearRegression()


# In[32]:


x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size = 0.20)
lr.fit(x_train,y_train)


# In[33]:


lr.coef_


# In[34]:


y_pred = lr.predict(x_test)
prediction = pd.DataFrame({'actual':y_test.flatten(), 'predicted': y_pred.flatten()})
prediction


# In[36]:


rsquare = lr.score(x_test,y_test)
rsquare


# In[ ]:




