import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, OPTICS  # Added OPTICS here
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
import streamlit as st
from sklearn.mixture import GaussianMixture  # <-- nou
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


class ExperimentalData:
    def __init__(self, df):
        self.df_all = df.copy()
        df_processed = df.copy()
        for col in df_processed.select_dtypes(include=[object]).columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        self.df = df_processed.select_dtypes(include=[np.number])
        self.X = self.df.values
        self.X_scaled = None
        self.y = None  # variabila țintă pentru supervised

    def set_target(self, target_column_name):
        if target_column_name in self.df_all.columns:
            self.y = self.df_all[target_column_name]
        else:
            print(f"Coloana '{target_column_name}' nu există în DataFrame.")

    def preprocess(self):
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def train_and_evaluate(self, test_size=0.2):
        if self.y is None:
            print("Setează coloana țintă cu set_target() înainte de antrenare.")
            return
        self.preprocess()
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=42)
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        return model

    def hierarchical_clustering(self, n_clusters, metric='euclidean'):
        linked = sch.linkage(self.X_scaled, method='ward', metric=metric)
        from scipy.cluster.hierarchy import fcluster
        labels = fcluster(linked, t=n_clusters, criterion='maxclust')
        return linked, labels

    def plot_dendrogram(self, linked, title='Dendrogram'):
        linked_array = np.asarray(linked, dtype=np.float64)
        plt.figure(figsize=(12, 8))
        sch.dendrogram(linked_array)
        plt.title(title)
        st.pyplot(plt)

    def kmeans_clustering(self, n_clusters):
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(self.X_scaled)
        centers = model.cluster_centers_
        return labels, centers

    def plot_kmeans(self, labels, centers, title='KMeans'):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        centers_pca = pca.transform(centers)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=100)
        plt.title(title)
        st.pyplot(plt)

    def dbscan_clustering(self, eps=3, min_samples=5):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(self.X_scaled)
        return labels

    def plot_dbscan_scatter(self, labels, title='DBSCAN Clusters'):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)

        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            class_member_mask = (labels == k)
            xy = X_pca[class_member_mask]
            if k == -1:
                plt.plot(xy[:, 0], xy[:, 1], 'k+', markersize=8, label='Noise')
            else:
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                         markeredgecolor='k', markersize=8, label=f'Cluster {k}')
        plt.title(title)
        plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        st.pyplot(plt)

    def gmm_clustering(self, n_components):
        model = GaussianMixture(n_components=n_components, random_state=42)
        labels = model.fit_predict(self.X_scaled)
        return labels

    def plot_gmm(self, labels, title='GMM Clustering'):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set2', s=30)
        plt.title(title)
        st.pyplot(plt)

    # New OPTICS clustering method
    def optics_clustering(self, min_samples=5, xi=0.02, min_cluster_size=5):
        model = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        labels = model.fit_predict(self.X_scaled)
        return labels

    def plot_optics_scatter(self, labels, title='OPTICS Clusters'):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)

        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            class_member_mask = (labels == k)
            xy = X_pca[class_member_mask]
            if k == -1:
                plt.plot(xy[:, 0], xy[:, 1], 'k+', markersize=8, label='Noise')
            else:
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                         markeredgecolor='k', markersize=8, label=f'Cluster {k}')
        plt.title(title)
        plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        st.pyplot(plt)

    def silhouette_score(self, labels):
        if len(set(labels)) > 1:
            return silhouette_score(self.X_scaled, labels)
        else:
            return -1

    def plot_dbscan_by_features(self, labels, feature_x='Sleep Hours', feature_y='Productivity',
                                title='DBSCAN Clusters by Features'):
        import seaborn as sns
        import pandas as pd

        # Reconstruct scaled DataFrame for reference to column names
        df_scaled = pd.DataFrame(self.X_scaled, columns=self.df.columns)
        df_scaled['Cluster'] = labels

        # Separate outliers
        outliers = df_scaled[df_scaled['Cluster'] == -1]
        clusters = df_scaled[df_scaled['Cluster'] != -1]

        # Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=clusters,
            x=feature_x,
            y=feature_y,
            hue='Cluster',
            palette='Set2',
            s=100,
            edgecolor='black'
        )

        plt.scatter(outliers[feature_x], outliers[feature_y], color='black', label='Outliers', s=20)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
    def run_all_methods(self, n_clusters):
        self.preprocess()

        # Hierarchical
        linked = sch.linkage(self.X_scaled, method='ward', metric='euclidean')
        linked = np.array(linked, dtype=np.float64)
        labels_hier = sch.fcluster(linked, t=n_clusters, criterion='maxclust')

        # KMeans
        labels_km, centers_km = self.kmeans_clustering(n_clusters)

        # DBSCAN
        labels_db = self.dbscan_clustering()

        # GMM
        labels_gmm = self.gmm_clustering(n_clusters)

        # OPTICS (new)
        labels_optics = self.optics_clustering()

        # Silhouette
        silh_hier = self.silhouette_score(labels_hier)
        silh_km = self.silhouette_score(labels_km)
        silh_db = self.silhouette_score(labels_db)
        silh_gmm = self.silhouette_score(labels_gmm)
        silh_optics = self.silhouette_score(labels_optics)

        return {
            'hierarchical': (labels_hier, linked),
            'kmeans': (labels_km, centers_km),
            'dbscan': labels_db,
            'gmm': labels_gmm,
            'optics': labels_optics,
            'silhouette': {
                'hierarchical': silh_hier,
                'kmeans': silh_km,
                'dbscan': silh_db,
                'gmm': silh_gmm,
                'optics': silh_optics
            }
        }

    def display_normalized_data(self):
        import pandas as pd
        if self.X_scaled is None:
            st.warning("Data has not been preprocessed yet. Run preprocess() first.")
            return

        df_scaled = pd.DataFrame(self.X_scaled, columns=self.df.columns)

        # Merge scaled numeric data with the original non-numeric columns
        non_numeric_cols = self.df_all.drop(columns=self.df.columns)
        df_combined = pd.concat([non_numeric_cols.reset_index(drop=True), df_scaled], axis=1)

        st.subheader("Normalized Data (with original non-numeric columns)")
        st.dataframe(df_combined)