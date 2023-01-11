import sys
import math
import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import cluster
import matplotlib.pyplot as plt

df = datasets.load_iris()
x=np.array([i for i in range(1,11)])
y=np.array([])
for K in range(1,11):
   y = np.append(y,cluster.KMeans(n_clusters=K,  init='k-means++',random_state=0).fit(df.data).inertia_)
plt.xlabel("K")
plt.ylabel("inertia")
plt.title("Find Optimal K Using Elbow Method")
plt.plot(x, y)
plt.annotate('Elbow',
             xy=(3, y[2]),
             xytext=(0.45, 0.45),
             textcoords='figure fraction',
             fontsize=12,
             arrowprops=dict(facecolor='black', shrink=0.1)
             )
plt.savefig("elbow.png")