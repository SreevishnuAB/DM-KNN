from src.visualise_data import visualise_dataset
from src.prepare_data import preprocess_data
from src.label_data import create_class_labels
from src.training import train_model
from src.inference import perform_inference

if __name__ == "__main__":
    print("Visualise dataset")
    print("=================\n")
    visualise_dataset()
    print("\nPlotting features")
    print("=================\n")
    preprocess_data()
    print("\nClustering")
    print("==========\n")
    create_class_labels()
    print("\nCreate models")
    print("=============\n")
    train_model()
    print("\nClassify new data point")
    print("=======================\n")
    annual_income = float(input("Annual Income:\n"))
    spending_score = float(input("Spending Score:\n"))
    perform_inference(annual_income, spending_score)