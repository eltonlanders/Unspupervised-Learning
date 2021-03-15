# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:34:58 2021

@author: elton
"""
import math
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, silhouette_score



# Calculating Euclidean Distance in Python
def dist(a, b):
    return math.sqrt(math.pow(a[0]-b[0],2) + math.pow(a[1]-b[1],2))

centroids = [ (2, 5), (8, 3), (4,5) ]
x = (0, 8)

# Calculating Euclidean Distance between x and centroid
centroid_distances =[]
for centroid in centroids:
    print("Euclidean Distance between x {} and centroid {} is {}".format(x, 
                                                centroid, dist(x, centroid)))
    centroid_distances.append(dist(x,centroid))



# Forming Clusters with the Notion of Distance
cluster_1_points =[ (0,8), (3,8), (3,4) ]

mean =[ (0+3+3)/3, (8+8+4)/3 ]
print(mean)



# K-means from Scratch – Part 1: Data Generation
X, y = make_blobs(n_samples=1500, centers=3, n_features=2, random_state=800)
centroids = [[-6,2],[3,-4],[-5,10]]

plt.scatter(X[:, 0], X[:, 1], s=50, cmap='tab20b') # plot without labels
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='tab20b') # plot with labels
plt.show()



# K-means from Scratch – Part 2: Implementing k-means
X[105:110] #choosing a subset

# Finds distances from each of 5 sampled points to all of the centroids
for x in X[105:110]:
    calcs = cdist(x.reshape([1,-1]), centroids).squeeze()
    print(calcs, "Cluster Membership: ", np.argmin(calcs)) 
    #argmin gives index of the min value along an axis useful for membership
#squeeze squeezes to a 1D array

def k_means(X, K): #k-means algorithm code
# Keep track of history so you can see K-Means in action
    centroids_history = []
    labels_history = []
    rand_index = np.random.choice(X.shape[0], K) 
    centroids = X[rand_index] # initialize random centroids
    centroids_history.append(centroids)
    while True:
# Euclidean distances are calculated for each point relative to centroids, 
# and then np.argmin returns the index location of the minimal distance to 
# which cluster a point is assigned to
        labels = np.argmin(cdist(X, centroids), axis=1)
        labels_history.append(labels)
        # Take mean of points within clusters to find new centroids:
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(K)])
        centroids_history.append(new_centroids)
        
        # If old centroids and new centroids no longer change, 
        # K-Means is complete and end, otherwise continue
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels, centroids_history, labels_history

centers, labels, centers_hist, labels_hist = k_means(X, 3)

history = zip(centers_hist, labels_hist)
for x, y in history:
    plt.figure(figsize=(4,3))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='tab20b');
    plt.scatter(x[:, 0], x[:, 1], c='red')    
    plt.show()



# Calculating the Silhouette Score which should ideally close to 1
np.random.seed(0)

seeds = pd.read_csv(r'tests/Seed_Data.csv')

X = seeds[['A','P','C','LK','WK','A_Coef','LKG']]

def k_means(X, K): #k-means algorithm code
# Keep track of history so you can see K-Means in action
    centroids_history = []
    labels_history = []
    rand_index = np.random.choice(X.shape[0], K) 
    centroids = X[rand_index] # initialize random centroids
    centroids_history.append(centroids)
    while True:
# Euclidean distances are calculated for each point relative to centroids, 
# and then np.argmin returns the index location of the minimal distance to 
# which cluster a point is assigned to
        labels = np.argmin(cdist(X, centroids), axis=1)
        labels_history.append(labels)
        # Take mean of points within clusters to find new centroids:
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(K)])
        centroids_history.append(new_centroids)
        
        # If old centroids and new centroids no longer change, 
        # K-Means is complete and end, otherwise continue
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels, centroids_history, labels_history

X_mat = X.values

centroids, labels, centroids_history, labels_history = k_means(X_mat, 3)

silhouette_score(X[['A','LK']], labels)



# Implementing k-means Clustering
np.random.seed(0)

seeds = pd.read_csv(r'tests/Seed_Data.csv')

X = seeds[['A','P','C','LK','WK','A_Coef','LKG']]
y = seeds['target']

# Bring back the function we created earlier
def k_means(X, K): #k-means algorithm code
# Keep track of history so you can see K-Means in action
    centroids_history = []
    labels_history = []
    rand_index = np.random.choice(X.shape[0], K) 
    centroids = X[rand_index] # initialize random centroids
    centroids_history.append(centroids)
    while True:
# Euclidean distances are calculated for each point relative to centroids, 
# and then np.argmin returns the index location of the minimal distance to 
# which cluster a point is assigned to
        labels = np.argmin(cdist(X, centroids), axis=1)
        labels_history.append(labels)
        # Take mean of points within clusters to find new centroids:
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(K)])
        centroids_history.append(new_centroids)
        
        # If old centroids and new centroids no longer change, 
        # K-Means is complete and end, otherwise continue
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels, centroids_history, labels_history

# Convert from Pandas dataframe to NumPy matrix
X_mat = X.values

# Run our Seeds matrix through the k_means function we created earlier
centroids, labels, centroids_history, labels_history = k_means(X_mat, 3)

# See how well our implementation of K-Means did
plt.scatter(X['A'], X['LK'])
plt.title('Wheat Seeds - Area vs Length of Kernel')
plt.show()

plt.scatter(X['A'], X['LK'], c=labels, cmap='tab20b')
plt.title('Wheat Seeds - Area vs Length of Kernel')
plt.show()

# Calculate Silhouette Score
silhouette_score(X[['A','LK']], labels)

