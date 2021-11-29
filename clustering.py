#-------------------------------------------------------------------------
# AUTHOR: Michael Tang
# FILENAME: clustering.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df.to_numpy()

silhouette_coefficient = []
bestK = 0
bestScore = 0
listOfK = []

#run kmeans testing different k values from 2 until 20 clusters
for k in range(2, 21):
    #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
    #      kmeans.fit(X_training)
    #--> add your Python code
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
    x = silhouette_score(X_training, kmeans.labels_)
    if bestScore < x:
         bestScore = x
         bestK = k
    silhouette_coefficient.append(x)
    listOfK.append(k)
#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
# Plot the metrics vs C_val
fig, ax = plt.subplots()
ax.set(xlabel='silhouette_coefficients',
          ylabel="k",
          title="k vs silhouette_coefficients")

ax.plot(silhouette_coefficient, listOfK, linestyle='-')

#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
df_test = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(df_test).reshape(1,len(df_test))[0]
#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here

#rung agglomerative clustering now by using the best value o k calculated before by kmeans
#Do it:
agg = AgglomerativeClustering(n_clusters=bestK, linkage='ward')
agg.fit(X_training)
#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
