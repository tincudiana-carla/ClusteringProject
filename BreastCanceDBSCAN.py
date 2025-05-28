import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# 1. Încarcă datele
data = pd.read_csv("BreastCancerDBSCANClustering.csv")

# 2. Preprocesare
labels_true = data['diagnosis'].map({'M': 1, 'B': 0}).values  # M=1, B=0
data_cleaned = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, errors='ignore')

# 3. Normalizează
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cleaned)

# 4. Reducere dimensionalitate pentru vizualizare
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# 5. Testăm pentru o gamă de valori eps
eps_values = np.linspace(0.5, 3.0, 11)
ari_scores = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(data_scaled)

    # ARI Score comparativ cu etichetele reale
    ari = adjusted_rand_score(labels_true, clusters)
    ari_scores.append(ari)

    plt.figure(figsize=(8, 5))

    for label in np.unique(clusters):
        mask = clusters == label
        if label == -1:
            plt.scatter(
                data_pca[mask, 0], data_pca[mask, 1],
                c='black', marker='x', label='Outliers (-1)', alpha=0.7
            )
        else:
            plt.scatter(
                data_pca[mask, 0], data_pca[mask, 1],
                label=f'Cluster {label}', alpha=0.7
            )

    plt.title(f'DBSCAN PCA (eps={eps:.2f}) | ARI: {ari:.2f}')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 6. Plot ARI în funcție de epsilon
plt.figure(figsize=(10, 6))
plt.plot(eps_values, ari_scores, marker='o', linestyle='-', color='blue')
plt.title('Adjusted Rand Index vs Epsilon (DBSCAN)')
plt.xlabel('Epsilon')
plt.ylabel('Adjusted Rand Index (ARI)')
plt.grid(True)
plt.tight_layout()
plt.savefig('ari_vs_epsilon.png')
plt.show()