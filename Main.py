import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from ExperimentalData import ExperimentalData

from clustering_methods import (hierarchical_clustering,
                                plot_dendrogram,
                                plot_pca,
                                dbscan_clustering,
                                plot_dbscan_scatter,
                                run_optics,
                                plot_optics_reachability,
                                plot_gmm, fit_gmm_select_components,
                                kmeans_clustering,
                                plot_kmeans,
                                fuzzy_cmeans_clustering,
                                plot_fuzzy)

# Date initiale
np.random.seed(42)
genes = ['Gene{}'.format(i + 1) for i in range(100)]
samples = ['Sample{}'.format(i + 1) for i in range(15)]
data, true_labels = make_blobs(n_samples=100, centers=3, cluster_std=4.5, n_features=15, random_state=42)
np.random.shuffle(data)
df = pd.DataFrame(data, index=genes, columns=samples)
df_csv = pd.read_csv("student_habits_performance.csv")
st.title("Clustering cu distanțe și comparare timp")

# Adaugă un meniu în topbar
metoda_clustering = st.selectbox(
    'Selectează metoda de clustering',
    ('Hierarchical Clustering', 'DBSCAN', 'OPTICS', 'EM' , 'KMeans', 'FuzzyCMeans', 'Experimental Data'),
    key='clustering_method_selectbox'
)

if metoda_clustering == 'Hierarchical Clustering':
    num_clusters = st.slider("Numărul de clustere", min_value=2, max_value=10, value=4, key='num_clusters_slider')

if metoda_clustering == 'DBSCAN':
    eps = st.slider('Epsilon (eps)', min_value=0.1, max_value=5.0, value=0.5, step=0.1, key='eps_slider')
    min_samples = st.slider('Min Samples', min_value=1, max_value=20, value=5, key='min_samples_slider')

if metoda_clustering == 'OPTICS':
    min_samples = st.slider('Min Samples', 1, 20, 5, key='optics_min_samples')
    xi = st.slider('Xi', 0.01, 0.2, 0.05, step=0.01, key='optics_xi')
    min_cluster_size = st.slider('Min Cluster Size (%)', 1, 50, 5, key='optics_mcs')

if metoda_clustering == 'EM':
    covariance_type = st.selectbox('Covariance Type', ('full', 'tied', 'diag', 'spherical'), key='cov_type_select')
if metoda_clustering == 'KMeans' or metoda_clustering == 'FuzzyCMeans':
    n_clusters = st.slider("Număr de clustere", 2, 10, 3)


if st.button("Rulează clustering"):
    if metoda_clustering == 'Hierarchical Clustering':

        start_time = time.time()
        dist_euclid, linked_euclid, clusters_euclid = hierarchical_clustering(df, num_clusters, metric='euclidean')
        time_euclid = time.time() - start_time

        start_time = time.time()
        dist_manhattan, linked_manhattan, clusters_manhattan = hierarchical_clustering(df, num_clusters, metric='cityblock')
        time_manhattan = time.time() - start_time

        # Afișare timp
        st.write(f"Timp clustering Euclidean: {time_euclid:.4f} secunde")
        st.write(f"Timp clustering Manhattan: {time_manhattan:.4f} secunde")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Clustere Euclidean (n={num_clusters})")
            df['Cluster Euclidean'] = clusters_euclid
            styled_df_eu = df.style.background_gradient(cmap='viridis')
            st.dataframe(styled_df_eu)

            # Dendrograma Euclidean
            fig_eu = plot_dendrogram(linked_euclid, df, num_clusters, 'Dendrograma - Distanța Euclidiană')
            st.pyplot(fig_eu)

            # PCA Euclidean
            fig_pca_eu = plot_pca(df.iloc[:, :-1], clusters_euclid, 'Reprezentare PCA - Euclidean')
            st.pyplot(fig_pca_eu)

        with col2:
            st.subheader(f"Clustere Manhattan (n={num_clusters})")
            df['Cluster Manhattan'] = clusters_manhattan
            styled_df_manh = df.style.background_gradient(cmap='viridis')
            st.dataframe(styled_df_manh)

            # Dendrograma Manhattan
            linked_manhattan = hierarchical_clustering(df, num_clusters, metric='cityblock')[1]
            fig_m = plot_dendrogram(linked_manhattan, df, num_clusters, 'Dendrograma - Distanța Manhattan')
            st.pyplot(fig_m)

            # PCA Manhattan
            fig_pca_m = plot_pca(df.iloc[:, :-1], clusters_manhattan, 'Reprezentare PCA - Manhattan')
            st.pyplot(fig_pca_m)


    elif metoda_clustering == 'DBSCAN':
            # Clustering cu DBSCAN
            data, true_labels = make_blobs(n_samples=100, centers=3, cluster_std=0.5, n_features=15, random_state=42)
            np.random.shuffle(data)
            df = pd.DataFrame(data, index=genes, columns=samples)
            start_time = time.time()
            clusters_dbscan = dbscan_clustering(df, eps=eps, min_samples=min_samples)
            time_dbscan = time.time() - start_time
            df['Cluster DBSCAN'] = clusters_dbscan
            st.write(f"Timp clustering DBSCAN: {time_dbscan:.4f} secunde")
            X = df.iloc[:, :-1].values
            labels = clusters_dbscan

            plt_obj = plot_dbscan_scatter(X, labels, 'Clustere DBSCAN (reducere PCA)')
            st.pyplot(plt_obj)


    elif metoda_clustering == "Experimental Data":

        exp = ExperimentalData(df_csv)

        exp.preprocess()

        exp.display_normalized_data()

        results = exp.run_all_methods(n_clusters=4)

        # Afișăm scorurile silhouette pentru toate metodele

        st.write("Silhouette scores:")

        for method, score in results['silhouette'].items():
            st.write(f"{method}: {score:.3f}")

        # 1. Clustering KMeans

        labels_km, centers_km = results['kmeans']

        exp.plot_kmeans(labels_km, centers_km, 'Clustering KMeans')

        # 2. Dendrograma Hierarchical

        labels_hier, linked = results['hierarchical']

        exp.plot_dendrogram(linked, 'Dendrograma Hierarchical')

        # 3. Clustere DBSCAN

        labels_db = results['dbscan']

        exp.plot_dbscan_scatter(labels_db, 'DBSCAN Clusters')

        # 4. Clustere OPTICS

        labels_optics = results['optics']

        exp.plot_optics_scatter(labels_optics, 'OPTICS Clusters')

        # 5. Clustere GMM

        labels_gmm = results['gmm']

        exp.plot_gmm(labels_gmm, 'Gaussian Mixture Model Clustering')

        # 6. Clustering după caracteristici (exemplu cu DBSCAN)

        labels = exp.dbscan_clustering()

        exp.plot_dbscan_by_features(labels, feature_x='sleep_hours', feature_y='netflix_hours')



