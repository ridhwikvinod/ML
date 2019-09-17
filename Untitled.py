#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lr=LinearRegression()

from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[2]:



boston=datasets.load_boston()
boston.feature_names
boston.keys()


# In[3]:


df=pd.DataFrame(data=boston.data,columns=boston.feature_names)
df.head()


# In[4]:



x=boston.data  
y=boston.target


# In[5]:



x_train,x_test,y_train,y_test = train_test_split(x,y)


# In[6]:


lr.fit(x_train,y_train)


# In[7]:



lr.coef_


# In[8]:


lr.intercept_


# In[9]:



pred=lr.predict(x_test)
print('cost per rooms:',pred[0]*1000)


# In[10]:



lr.score(x_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:




