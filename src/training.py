import pickle
from matplotlib.pyplot import axis
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


def train_knn_model(clusters, X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=clusters)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"KNN Confusion matrix:\n{metrics.confusion_matrix(y_test, y_pred)}\n")
    print(f"KNN Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    knn_model = open(".\\models\\spend_model_knn.pkl", "wb")
    pickle.dump(knn, knn_model)


def train_ensemble_model(X_train, X_test, y_train, y_test):
    models = []
    models.append(("knn1", KNeighborsClassifier(n_neighbors=1)))
    models.append(("knn3", KNeighborsClassifier(n_neighbors=3)))
    models.append(("knn5", KNeighborsClassifier(n_neighbors=5)))
    models.append(("knn7", KNeighborsClassifier(n_neighbors=7)))
    models.append(("knn9", KNeighborsClassifier(n_neighbors=9)))
    ensemble = VotingClassifier(estimators=models, voting="hard")
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    print(f"Ensemble Confusion matrix:\n{metrics.confusion_matrix(y_test, y_pred)}\n")
    print(f"Ensemble Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    ensemble_model = open(".\\models\\spend_model_ensemble.pkl", "wb")
    pickle.dump(ensemble, ensemble_model)
    

def train_model(test_set_size=0.3, knn_clusters=5):
    data = pd.read_csv(".\\datasets\\mall_data_labelled.csv")
    labels = data["Label"]
    features = data[["Annual Income", "Spending Score"]]
    print(features.head())
    print(f"Train:test = {1-test_set_size}:{test_set_size}")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_set_size, random_state=2)
    train_knn_model(knn_clusters, X_train, X_test, y_train, y_test)
    train_ensemble_model(X_train, X_test, y_train, y_test)
    X_train_with_label = X_train.join(y_train)
    X_train_with_label.to_csv(".\\datasets\\mall_data_train.csv")


if __name__ == "__main__":
    train_model()