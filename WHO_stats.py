#!/usr/bin/env python
# coding: utf-8

# In[119]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import norm


# In[120]:


df = pd.read_csv('F:\WHO_stats.csv')
df.dtypes 


# In[122]:


#To find out whether all the columns have values
df.count()


# In[123]:


#total no of rows and columns
df.shape


# In[125]:


#rows containing duplicate data
duplicate_rows = df[df.duplicated()]
print("no of duplicate rows:", duplicate_rows.shape)
#there are no duplicate values


# In[126]:


#Detect if there is any missing data
df.isnull().sum()


# In[128]:


#fill the missing data as 0
df.fillna(value=0, inplace = True)


# In[129]:


#check if there is any missing data now after filling the value as 0
df.isnull().sum()


# In[130]:


#change the object datatype to int datatype
df['population'] = df['population'].astype('int64')
df['suicides_no'] = df['suicides_no'].astype('int64')


# In[131]:


df.dtypes


# In[46]:


#visualising the outliers
sb.boxplot(x=df['suicides_no'])


# In[132]:


# Find the interquartile range
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[133]:


#Eliminate the outliers - data which doesn't falls between this range
df = df[~((df < (Q1-1.5 * IQR)) | (df > (Q3+1.5 * IQR))).any(axis=1)]
df.shape


# In[134]:


# barplot showing the suicides number classified based on sex for the 10 countries with largest population
df_suicides = df.nlargest(10,'population')
print(df_suicides)
plt.figure(figsize = [8,7])
sb.barplot(x='country', y='suicides_no', hue='sex', data=df_suicides)


# In[135]:


#plot a histogram which shows the distribution of values in suicide_no column
bin_edges = np.arange(0, df['suicides_no'].max(), 4)
plt.hist(data = df, x = 'suicides_no', bins = bin_edges)
plt.axvline(df['suicides_no'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.axvline(df['suicides_no'].median(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(df['suicides_no'].mean()*1.1, max_ylim*0.4, 'Average- {:.2f}'.format(df['suicides_no'].mean()))
plt.text(df['suicides_no'].median()*1.1, max_ylim*0.8, 'Median- {:.2f}'.format(df['suicides_no'].median()))


# In[136]:


# Plot a scatter plot which shows the correlation between suicides and population
fig, ax = plt.subplots(figsize=(15,7))
ax.scatter(df['population'], df['suicides_no'])
ax.set_xlabel('population')
ax.set_ylabel('No of suicides')
plt.show()


# In[ ]:




