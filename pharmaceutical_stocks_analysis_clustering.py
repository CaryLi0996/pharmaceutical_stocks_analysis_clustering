#!/usr/bin/env python
# coding: utf-8

# # Analyze pharmaceutical stocks with clustering

# In[ ]:


# import libraries and packages

get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates


# In[ ]:


# import dataset
pharma_df = pd.read_csv('Pharmaceuticals.csv')
pharma_df.set_index('Name', inplace=True)
pharma_df

# only use the important numeric variables from 1-9:
numeric = ['Market_Cap', 'Beta', 'PE_Ratio', 'ROE','ROA', 'Asset_Turnover', 'Leverage', 'Rev_Growth', 'Net_Profit_Margin' ]

# conversion of integer data to float will avoid a warning when applying the scale function
pharma_df[numeric] = pharma_df[numeric].apply(lambda x: x.astype('float64'))
pharma_df.info()


# In[ ]:


num_df = pharma_df[numeric]
num_df


# In[ ]:


#Euclidean distance used as the most popular method (dissimilarity matrix)

d = pairwise.pairwise_distances(num_df, metric='euclidean')
pd.DataFrame(d, columns=num_df.index, index=num_df.index)


# In[ ]:


# Normalize data
# pandas uses sample standard deviation/ n-1 so this was used
num_df_norm = (num_df - num_df.mean())/num_df.std()
num_df_norm

#num_df_norm = num_df.apply(preprocessing.scale, axis=0)
#num_df_norm


# In[ ]:


#Silhoutte Score
from yellowbrick.cluster import SilhouetteVisualizer
for i in [2, 3, 4, 5, 6]:

  km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
  visualizer = SilhouetteVisualizer(km)
  visualizer.fit(num_df_norm)
  plt.show()


# The value of 4 for n_clusters looks at first glance to be the optimal one. The silhouette score for each cluster is above average silhouette scores.  The thickness of the silhouette plot representing each cluster also is a deciding point. However, there's a negative fluctuation in the silhouette score on one of the clusters therefore 3 clusters seem to be the next best choice.

# In[ ]:


#Check silhoutte score plot for each k

from sklearn.metrics import silhouette_samples,silhouette_score
import numpy as np
sillhoute_scores = []

n_cluster_list = np.arange(2,10).astype(int)

for n_cluster in n_cluster_list:
    
    kmeans = KMeans(n_clusters=n_cluster,random_state=0)
    cluster_found = kmeans.fit_predict(num_df_norm)
    sillhoute_scores.append(silhouette_score(num_df_norm, kmeans.labels_))

silbycluster = pd.DataFrame(sillhoute_scores,n_cluster_list)
silbycluster.plot()
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.show()


# In[ ]:


#Elbow Plot (elbow is at 2 or 3)

inertia = []
for n_clusters in range(1, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(num_df_norm)
    inertia.append(kmeans.inertia_ / n_clusters)
inertias = pd.DataFrame({'n_clusters': range(1, 10), 'inertia': inertia})
ax = inertias.plot(x='n_clusters', y='inertia')
plt.xlabel('Number of clusters(k)')
plt.ylabel('Average Within-Cluster Squared Distances')
plt.ylim((0, 1.1 * inertias.inertia.max()))
ax.legend().set_visible(False)
plt.show()


# In[ ]:


#Check with Gap Statistic 

def optimalK(data, nrefs=2, maxClusters=10):
    """
    Calculates KMeans optimal K using Gap Statistic 
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
        
# Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
    
# For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k,random_state=0)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
            
# Fit cluster to original data and create dispersion
        km = KMeans(k,random_state=0)
        km.fit(data)
        
        origDisp = km.inertia_
        
# Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
    
# Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    return (gaps.argmax() + 1, resultsdf)
score_g, df = optimalK(num_df_norm, nrefs=2, maxClusters=15)
plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('Gap Statistic');
plt.title('Gap Statistic vs. K');


# In[ ]:


# Fit a k-Means clustering with k=3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(num_df_norm)

# Cluster membership
memb = pd.Series(kmeans.labels_, index=num_df_norm.index)
print('\033[1m'+'k-Means cluster membership:'+'\033[0m')
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[ ]:


centroids = pd.DataFrame(kmeans.cluster_centers_, 
   columns=num_df_norm.columns)
pd.set_option('precision', 3)
centroids


# In[ ]:


# WITHIN CLUSTERS: distances of each record to the cluster centers
distances = kmeans.transform(num_df_norm)

# find closest cluster for each record
minSquaredDistances = distances.min(axis=1) ** 2

# combine with cluster labels into a data frame
df = pd.DataFrame({'squaredDistance': minSquaredDistances, 
   'cluster': kmeans.labels_}, index=num_df_norm.index)

# group by cluster and print information 
for cluster, data in df.groupby('cluster'):
   count = len(data)
   withinClustSS = data.squaredDistance.sum()
   print(f'Cluster{cluster}({count} members):{withinClustSS:.2f} within cluster')


# In[ ]:


#Pairwise distance from cluster centers

print(pd.DataFrame(pairwise.pairwise_distances(kmeans.cluster_centers_, metric='euclidean')))


# In[ ]:


#Sum of Pairwise distance for each clusters

print(pd.DataFrame(pairwise.pairwise_distances(kmeans.cluster_centers_, metric='euclidean')).sum(axis=0))


# In[ ]:


# visualization of the different clusters

centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]
fig = plt.figure(figsize=(10, 6))
plt.figure(figsize=(10,6))
fig.subplots_adjust(right=3)
ax = parallel_coordinates(centroids, class_column='cluster', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
plt.xlim(0.0,8.5)


# In[ ]:


#Convert Categorical to numerical

pharma_df = pharma_df.drop(columns='Symbol')
new_df = pd.get_dummies(pharma_df, prefix_sep='_', drop_first=True)
new_df


# In[ ]:


#Run clustering with new data with categorical variables
# Normalized distance
new_df_num_norm = new_df.apply(preprocessing.scale, axis=0)

# Fit a k-Means clustering with k=3 clusters
new_kmeans = KMeans(n_clusters=3, random_state=0).fit(new_df_num_norm)

# Cluster membership
new_memb = pd.Series(new_kmeans.labels_, index=new_df_num_norm.index)
print('\033[1m'+'k-Means cluster membership:'+'\033[0m')
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# **With categorical variables, cluster membership remains unchanged.**

# In[ ]:


#Create a column with the assigned clusters
cluster0 = ['Aventis', 'Elan Corporation, plc', 'Medicis Pharmaceutical Corporation', 'Watson Pharmaceuticals, Inc.']
cluster1 = ['Abbott Laboratories', 'AstraZeneca PLC', 'Bristol-Myers Squibb Company', 'Eli Lilly and Company', 'GlaxoSmithKline plc', 'Johnson & Johnson', 'Merck & Co., Inc.', 'Novartis AG', 'Pfizer Inc', 'Schering-Plough Corporation', 'Wyeth']
cluster2 = ['Allergan, Inc.', 'Amersham plc', 'Bayer AG', 'Chattem, Inc', 'IVAX Corporation', 'Pharmacia Corporation']


pharma_df['Cluster'] = num_df.index
pharma_df.loc[cluster0,'Cluster'] = 0 
pharma_df.loc[cluster1,'Cluster'] = 1 
pharma_df.loc[cluster2,'Cluster'] = 2


#look at values of each cluster:
pharma_df.sort_values(by=['Cluster'])


# In[ ]:


#Average Distance: Hierarchical Clustering to examine the cluster groups to validate k-means clusters for ourselves, not for the questions
Z = linkage(num_df_norm, method='average')

fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(bottom=0.23)
plt.title('Hierarchical Clustering Dendrogram (Average linkage)')
plt.xlabel('Company')
dendrogram(Z, labels=num_df_norm.index, color_threshold=4.8)
plt.axhline(y=4.8, color='black', linewidth=0.5, linestyle='dashed')
plt.show()

