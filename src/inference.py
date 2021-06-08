import pickle
import pandas as pd

def perform_inference_knn(data_point, dataset):
    model = pickle.load(open(".\\models\\spend_model_knn.pkl", "rb"))
    print(f"Predicted class: {model.predict([data_point])}")
    nearest_neighbors = model.kneighbors([data_point], n_neighbors=3)
    print("Nearest neighbors")
    for i in range(len(nearest_neighbors[1][0])):
        print(dataset.loc[nearest_neighbors[1][0][i]][["Annual Income", "Spending Score"]])
        print(f"Distance: {nearest_neighbors[0][0][i]}\n")


def perform_inference_ensemble(data_point):
    model = pickle.load(open(".\\models\\spend_model_ensemble.pkl", "rb"))
    print(f"Predicted class: {model.predict([data_point])}")
    

def perform_inference(annual_income, spending_score):
    data = pd.read_csv(".\\datasets\\mall_data_train.csv")
    data.rename(columns={'Unnamed: 0':'Index'}, inplace=True)
    perform_inference_knn([annual_income, spending_score], data)
    perform_inference_ensemble([annual_income, spending_score])
    

if __name__ == "__main__":
    annual_income = float(input("Annual income: "))
    spending_score = float(input("Spending score: "))
    perform_inference(annual_income, spending_score)
