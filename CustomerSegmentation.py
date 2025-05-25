import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 1. Încarcă datele
data = pd.read_csv("ierarhicclusterdata.csv")
data_cleaned = data.drop(['CUST_ID'], axis=1)
data_cleaned.fillna(data_cleaned.mean(), inplace=True)

# 2. Normalizează
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cleaned)

# 3. PCA pentru vizualizare
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# 4. Metode linkage
linkage_methods = ['single', 'complete', 'average']
num_clusters = 4  # Setează numărul de clustere

for method in linkage_methods:
    # 5. Linkage și dendrogramă
    linked = linkage(data_scaled, method=method)

    # Plot dendrogramă într-o figură separată
    plt.figure(figsize=(8, 5))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title(f"Dendrogramă - {method.capitalize()} Linkage")
    plt.xlabel("Clienți (trunchiat)")
    plt.ylabel("Distanță")
    plt.tight_layout()
    plt.show()

    # 6. Extrage etichetele clustere
    cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')

    # Plot clustere PCA într-o figură separată
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=cluster_labels, palette='tab10', legend='full')
    plt.title(f"Clustere PCA - {method.capitalize()} Linkage")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()


