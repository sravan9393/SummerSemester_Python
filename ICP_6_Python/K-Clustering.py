import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('CC.csv')
x = dataset.iloc[:,[1,15]]
y = dataset.iloc[:,-1]
print(x.shape, y.shape)
# see how many samples we have of each species
print(dataset["TENURE"].value_counts())

import matplotlib.pyplot as plt
import numpy as np
correlation = dataset.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(dataset.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(dataset.columns)
ax.set_yticklabels(dataset.columns)
plt.show()


sns.FacetGrid(dataset, hue="TENURE", size=4).map(plt.scatter, "BALANCE_FREQUENCY", "CREDIT_LIMIT").add_legend()
sns.FacetGrid(dataset, hue="TENURE", size=4).map(plt.scatter, "BALANCE", "MINIMUM_PAYMENTS").add_legend()
plt.show()

## Null values
nulls = pd.DataFrame(x.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

## Replacing null values with mean values
x = x.select_dtypes(include=[np.number]).interpolate().dropna()

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

from sklearn.cluster import KMeans
nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)

##elbow method to know the number of clusters
wcss = []
for i in range(1,7):
     kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
     kmeans.fit(x)
     wcss.append(kmeans.inertia_)

plt.plot(range(1,7),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
