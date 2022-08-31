import matplotlib . pyplot as plt
from sklearn . cluster import KMeans
import numpy as np
import pandas as pd
import os
pd.options.mode.chained_assignment = None

goog_path = os.path.abspath('GOOG_weekly_return_volatility.csv')
df_goog = pd.read_csv(goog_path)
df_googvol = df_goog[df_goog.Year.isin([2019])]
df_googvol_2yrs = df_goog[df_goog.Year.isin([2019,2020])]

X = df_googvol_2yrs [["mean_return", "volatility"]]
y = df_googvol_2yrs["Label"]


print('############## Q1  ################')

# Part a) where k = 3
def kmeanscluster(X,n):
    kmeans_classifier = KMeans ( n_clusters = n )
    y_means = kmeans_classifier . fit_predict (X)
    centroids = kmeans_classifier . cluster_centers_
    return y_means , centroids

y_means,centroids = kmeanscluster(X,3)
print(y_means)
print(centroids)

# Part b) to compute the best k by computing distortion vs K
y_kmeans = []
inertia_list = []
for k in range (1 ,9):
    kmeans_classifier = KMeans ( n_clusters =k)
    y_kmeans = kmeans_classifier . fit_predict (X)     
    inertia = kmeans_classifier . inertia_
    inertia_list . append ( inertia )
fig , ax = plt. subplots (1, figsize =(7 ,5))
plt . plot ( range (1, 9) , inertia_list , marker ='o',color ='green')
plt . legend ()
plt . xlabel ('number of clusters : k')
plt . ylabel ('inertia')
plt . tight_layout ()
plt . show ()

print('############## Q2 , Q3 ################')
#Computing percentage of green and red weeks in each cluster.
y_means_5,centroids_5 = kmeanscluster(X,5)
df_googvol_2yrs['Cluster'] = y_means_5
# df_googvol_2yrs.to_csv('df_googvol_2yrs.csv',index =False)
counts = df_googvol_2yrs.groupby(['Cluster', 'Label']).agg({'volatility': 'count'})
counts = counts.rename({'volatility': 'Counts'}, axis='columns')
print(counts)
state_pcts = counts.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
state_pcts = state_pcts.rename({'Counts': 'Percentage%'}, axis='columns')
print(state_pcts)

