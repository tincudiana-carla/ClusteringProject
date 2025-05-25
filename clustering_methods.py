import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.mixture import GaussianMixture

def hierarchical_clustering(df, num_clusters, metric='euclidean'):
    dist_matrix = pdist(df.values, metric=metric)
    linked = linkage(dist_matrix, method='ward')
    clusters = fcluster(linked, t=num_clusters, criterion='maxclust')
    return dist_matrix, linked, clusters

def plot_dendrogram(linked, df, num_clusters, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    level = linked[-(num_clusters - 1), 2]
    dendrogram(linked, labels=df.index, ax=ax, color_threshold=level)
    ax.axhline(y=level, color='r', linestyle='--', linewidth=1)
    ax.set_title(title)
    return fig

def plot_pca(df, clusters, title):
    coords = PCA(n_components=2).fit_transform(df)
    pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
    pca_df['Gene'] = df.index
    pca_df['Cluster'] = clusters
    fig, ax = plt.subplots()
    for c in np.unique(clusters):
        subset = pca_df[pca_df['Cluster'] == c]
        ax.scatter(subset['PC1'], subset['PC2'], label=f'Cluster {c}')
    ax.set_title(title)
    ax.legend()
    return fig

def dbscan_clustering(df, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(df.values)
    return clusters


def plot_dbscan_scatter(X, labels, title):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)

    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        if k == -1:
            plt.plot(xy[:, 0], xy[:, 1], 'k+', markersize=6, label='Noise')
        else:
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6, label=f'Cluster {k}')
    plt.title(title)
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    return plt

def run_optics(X, min_samples=5, xi=0.05, min_cluster_size=0.05):
    model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    model.fit(X)
    labels = model.labels_
    reachability = model.reachability_[model.ordering_]
    order = model.ordering_
    return labels, reachability, order

def plot_optics_reachability(reachability, ordering, title):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(reachability)), reachability, 'b.')
    plt.xlabel('Puncte în ordinea de vizitare')
    plt.ylabel('Distanță de reachability')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return plt

def em_clustering(X, n_components=3):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    return labels, gmm.means_, gmm.covariances_

def em_clustering_with_covariance(X, n_components=3, covariance_type='full'):
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    return labels, gmm.means_, gmm.covariances_

def plot_gmm(X, labels, means, covariances, title, covariance_type):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        data = X[labels == k]
        ax.scatter(data[:, 0], data[:, 1], s=40, color=col, label=f'Cluster {k}')

        # Gestione covarianță
        cov = None
        if covariance_type == 'full':
            cov = covariances[k][:2, :2]
        elif covariance_type == 'tied':
            cov = covariances[:2, :2]
        elif covariance_type == 'diag':
            cov = np.diag(covariances[k][:2])
        elif covariance_type == 'spherical':
            cov_scalar = covariances[k]
            cov = np.eye(2) * cov_scalar

        try:
            v, w = np.linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = np.degrees(angle)
            ell = Ellipse(xy=means[k][:2], width=v[0], height=v[1], angle=angle, alpha=0.2, color=col)
            ax.add_artist(ell)
        except np.linalg.LinAlgError:
            pass

    ax.set_title(title)
    ax.legend()
    return fig

def fit_gmm_select_components(X, max_components=10, criterion='bic'):
    models = []
    scores = []
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(X)
        models.append(gmm)
        score = gmm.bic(X) if criterion=='bic' else gmm.aic(X)
        scores.append(score)
    best_n = np.argmin(scores) + 1
    best_gmm = models[best_n - 1]
    labels = best_gmm.predict(X)
    return labels, best_gmm.means_, best_gmm.covariances_, best_n

def kmeans_clustering(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model.cluster_centers_

def plot_kmeans(X, labels, centers, title):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=50)
    ax.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=200)
    ax.set_title(title)
    return fig

def fuzzy_cmeans_clustering(X, n_clusters=3):
    import skfuzzy as fuzz
    # Transformați datele pentru scikit-fuzzy (shape trebuie să fie (features, samples))
    X_tf = X.T
    # Efectuează clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_tf, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)
    labels = np.argmax(u, axis=0)  # cel mai probabil cluster pentru fiecare punct
    return labels, cntr, u

def plot_fuzzy(X, labels, u, title):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', s=50)
    ax.set_title(title)
    return fig