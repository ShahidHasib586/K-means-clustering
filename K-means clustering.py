#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')


# In[3]:


dataset


# In[4]:


X = dataset.iloc[:, [3,4]].values


# In[5]:


X


# In[11]:


#using the elbow method to find the optional number of clusters
from sklearn.cluster import KMeans
wcss =[]
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state =0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[13]:


#applying the kmeans to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state =0)
y_kmeans = kmeans.fit_predict(X)


# In[14]:


y_kmeans


# In[18]:


#visulising the result 
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans == 0, 1], s =100, c ='red', label = 'Careful')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans == 1, 1], s =100, c ='blue', label = 'Standard')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans == 2, 1], s =100, c ='green', label = 'Target')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans == 3, 1], s =100, c ='yellow', label = 'careless')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans == 4, 1], s =100, c ='cyan', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s =300, c ='magenta', label = 'Centroids ')
plt.title('Cluster of Clients')
plt.xlabel('Anual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




