#!/usr/bin/env python
# coding: utf-8

# In[29]:


from sklearn import datasets
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# In[20]:


dataset = datasets.load_iris()
x,y=dataset.data,dataset.target


# In[21]:


model = GaussianNB()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[23]:


model.fit(x_train,y_train)


# In[25]:


y_predict=model.predict(x_test)


# In[30]:


print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_predict))
print('Mean squared error:',metrics.mean_squared_error(y_test,y_predict))
print('Root Mean squared error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))


# In[26]:


print(metrics.classification_report(y_test,y_predict))


# In[27]:


print(metrics.confusion_matrix(y_test, y_predict))

