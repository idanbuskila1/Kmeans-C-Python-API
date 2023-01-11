import sys
import math
import numpy as np
import pandas as pd
import mykmeanssp as kmeans

# initializes K,N,epsilon, datapoints and validates input 

def initialize():
    argc = len(sys.argv)
    K = 0
    iter = 0
    epsilon = 0
    dimension = 0
    file1_index = -1

    if argc == 6:  # iteration number was passed
        file1_index = 4
        try:
            iter = int(sys.argv[2])
        except ValueError:
            print("Invalid maximum iteration!")
            raise SystemExit

    elif argc == 5:  # iteration number wasnt passed
        file1_index = 3
        iter = 300 #default value

    else:  # wrong number of args
        print("An Error Has Occured")
        raise SystemExit

    #initialize K,epsilon    
    try:
        K = int(sys.argv[1])
    except ValueError:
        print("Invalid number of clusters!")
        raise SystemExit
    try:
        epsilon = float(sys.argv[file1_index-1])
    except ValueError:
        print("Invalid epsilon!")
        raise SystemExit

    #initialize datapoints from files
    file1= pd.read_csv(sys.argv[file1_index])
    header = ["d"+str(i) for i in range(file1.shape[1])]
    header[0] = "index"
    file1 = pd.read_csv(sys.argv[file1_index],names=header)
    file2 = pd.read_csv(sys.argv[file1_index + 1])
    header = ["d"+str(i+file1.shape[1]-1) for i in range(file1.shape[1])]
    header[0] = "index"
    file2 = pd.read_csv(sys.argv[file1_index + 1], names=header)
    datapoints = pd.merge(file1, file2, on='index')
    datapoints = datapoints.sort_values('index')
    N= datapoints.shape[0]
    dimension = datapoints.shape[1]-1

    #validate data
    if K <= 1 or K >= N:
        print("Invalid number of clusters!")
        raise SystemExit
    if iter <= 1 or iter >= 1000:
        print("Invalid maximum iteration!")
        raise SystemExit
    return (K, iter,epsilon, datapoints,N, dimension)

def distance(p, q):
    x=np.subtract(p,q)
    return np.sqrt(np.sum(np.multiply(x,x)))

def updateDistances(cur_distances, centroids_indexes, datapoints):
    index = centroids_indexes[-1]
    new_centroid = np.array(datapoints.loc[datapoints['index'] == index])[0,1:]
    for i in range(cur_distances.shape[0]):
        j = cur_distances[i, 1]
        point = np.array(datapoints.loc[datapoints['index'] == j])[0,1:]
        dist = distance(point, new_centroid)
        if cur_distances[i,0]>dist:
            cur_distances[i,0] = dist


def addCentroid(min_distances , datapoints):
    dist_sum = np.sum(min_distances[:,0])
    distribution = np.array([float(min_distances[i,0] /dist_sum) for i in range(min_distances.shape[0])])
    indexes = np.array(datapoints['index'])
    centroid_index = np.random.choice(indexes, p=distribution)
    return centroid_index

def initialize_centroids(K, datapoints):
    centroids_indexes = np.array([])
    min_distances = np.array([[math.inf, datapoints.iloc[i,0]] for i in range(datapoints.shape[0])])

    # randomly choose first centroid
    np.random.seed(0)
    rand_index = np.random.choice(np.array(datapoints.iloc[:, 0]))
    centroids_indexes = np.append(centroids_indexes, rand_index)
    updateDistances(min_distances, centroids_indexes, datapoints)

    #choose another K-1 centroids by kmeans++ algorithm
    for i in range(1,K):
        centroids_indexes = np.append(centroids_indexes, addCentroid(min_distances, datapoints))
        updateDistances(min_distances, centroids_indexes, datapoints)
    #return centroids
    centroids = np.array(datapoints.loc[datapoints['index'] == centroids_indexes[0]])[0, 1:]
    for i in range(1,K):
        centroid = np.array(datapoints.loc[datapoints['index'] == centroids_indexes[i]])[0, 1:]
        centroids = np.vstack([centroids,centroid])
    return centroids,centroids_indexes

def printRow(row, N, format):
    for i in range(N-1):
        print(format % row[i], end=",")
    print(format % row[N-1])


### MAIN ###
K,iter,epsilon,datapoints,N, dimension = initialize()
centroids, indexes = initialize_centroids(K,datapoints)
#find centroids with c module
centroids = np.array(kmeans.fit(K, N, iter, dimension,epsilon,np.array(datapoints.values)[:,1:].tolist(),centroids.tolist()))

###OUTPUT###
printRow(indexes,K,"%d")
for i in range(centroids.shape[0]): printRow(centroids[i], dimension, "%.4f")
