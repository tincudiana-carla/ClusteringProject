import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('covid_19_indonesia_time_series_all.csv')

df = df[df['Location ISO Code'].str.startswith('ID-')]

df_grouped = df.groupby('Location').agg({
    'New Cases': 'mean',
    'New Deaths': 'mean',
    'New Active Cases': 'max'
}).rename(columns={
    'New Cases': 'Avg New Cases',
    'New Deaths': 'Avg New Deaths',
    'New Active Cases': 'Max Active Cases'
})

df_grouped.fillna(0, inplace=True)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_grouped)

# 6. Determinarea numărului optim de clustere (k)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# 7. Plot metoda Elbow
plt.figure(figsize=(8,6))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Metoda Elbow pentru determinarea numărului optim de clustere')
plt.xlabel('Numărul de clustere')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

k_opt = 3
kmeans = KMeans(n_clusters=k_opt, random_state=42)
df_grouped['Cluster'] = kmeans.fit_predict(df_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(df_grouped['Avg New Cases'], df_grouped['Max Active Cases'],
                      c=df_grouped['Cluster'], cmap='viridis', s=50)
plt.xlabel('Media cazurilor noi')
plt.ylabel('Maxim cazuri active')
plt.title('Clustering K-Means pe cazuri noi medii și cazuri active maxime')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()

