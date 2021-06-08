from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_class_labels(clusters=5):
    mall_data_scaled = pd.read_csv(".\\datasets\\mall_data_scaled.csv")
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans = kmeans.fit(mall_data_scaled)
    print(f"Labels: {kmeans.labels_}")
    cluster_data = [{"cluster": i,"cluster center": kmeans.cluster_centers_[i], "no. of records": len(list(np.where(kmeans.labels_ == i)[0]))} for i in range(kmeans.n_clusters)]
    print("Cluster data:")
    for cluster in cluster_data:
        print(cluster)

    features_scaled = np.array(mall_data_scaled)
    mall_data_scaled["cluster"] = kmeans.labels_
    plt.scatter(features_scaled[kmeans.labels_==0,0],features_scaled[kmeans.labels_==0,1],s=80,c='magenta',label='High income, Low spend')
    plt.scatter(features_scaled[kmeans.labels_==1,0],features_scaled[kmeans.labels_==1,1],s=80,c='yellow',label='Low income, High spend')
    plt.scatter(features_scaled[kmeans.labels_==2,0],features_scaled[kmeans.labels_==2,1],s=80,c='green',label='High income, High spend')
    plt.scatter(features_scaled[kmeans.labels_==3,0],features_scaled[kmeans.labels_==3,1],s=80,c='cyan',label='Moderate income, Moderate spend')
    plt.scatter(features_scaled[kmeans.labels_==4,0],features_scaled[kmeans.labels_==4,1],s=80,c='burlywood',label='Low income, Low spend')
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker = "o", alpha = 0.9,s=250,c='red',label='Centroids')
    plt.title('Customer cluster')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.show()
    labels_list = [('High', 'Low'), ('Low', 'High'), ('High', 'High'), ('Moderate', 'Moderate'), ('Low', 'Low')]
    mall_data_labelled = pd.read_csv(".\\datasets\\mall_data.csv")
    mall_data_labelled["Label"] = np.array([f"{labels_list[_label][0]} income, {labels_list[_label][1]} spend" for _label in kmeans.labels_])
    print(mall_data_labelled.head())
    mall_data_labelled.to_csv(".\\datasets\\mall_data_labelled.csv", index=False)


if __name__ == "__main__":
    create_class_labels()