import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def preprocess_data(max_clusters=10):
    mall_data = pd.read_csv(".\\datasets\\mall_data.csv")
    features = np.array(mall_data)
    features_scaled = MinMaxScaler().fit_transform(features)
    df = pd.DataFrame(data=features_scaled, columns=["Annual Income", "Spending Score"])
    df.to_csv(".\\datasets\\mall_data_scaled.csv", index=False)
    # k-value selection using elbow method
    inertia = []
    k_range = range(2, max_clusters)
    for i in k_range:
        km = KMeans(n_clusters=i)
        km = km.fit(features_scaled)
        inertia.append(km.inertia_) # Sum of squared distances of samples to their closest cluster center

    # plotting elbow graph
    plt.plot(k_range, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of squared distances')
    plt.title('Optimal k')
    plt.show()


if __name__ == "__main__":
    preprocess_data()