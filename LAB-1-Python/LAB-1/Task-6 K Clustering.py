# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

# Loading the dataset
dataset = pd.read_csv('./datasets/College.csv')
print(dataset.dtypes)


# splitting the features and class
x = dataset.iloc[:,[1,16]]
y = dataset.iloc[:,-1]
print(x.shape, y.shape)

# see how many samples we have of each species
print(dataset["Terminal"].value_counts())

## Printing the count of Null values
nulls = pd.DataFrame(x.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Plotting scatter graph for columns
sns.FacetGrid(dataset, hue="Terminal", size=4).map(plt.scatter, "S.F.Ratio", "Expend").add_legend()
sns.FacetGrid(dataset, hue="Terminal", size=4).map(plt.scatter, "Apps", "Accept").add_legend()
plt.show()


## Replacing null values with mean values
x = x.select_dtypes(include=[np.number]).interpolate().dropna()

# Standardising the dataset
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

# Applying K-Means Clustering on the dataset
from sklearn.cluster import KMeans
nclusters = 2 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("Silhoutte Score: " + str(score))

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