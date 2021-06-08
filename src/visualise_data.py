import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualise_dataset():
    mall_data = pd.read_csv(".\\datasets\\Mall_Customers.csv")
    mall_data = mall_data.rename(columns={"Annual Income (k$)": "Annual Income", "Spending Score (1-100)": "Spending Score"})
    print(mall_data[["Age", "Annual Income", "Spending Score"]].describe())
    sorted_mall_data = mall_data.groupby(['Gender'])['Spending Score'].median().sort_values()
    sns.boxplot(x=mall_data['Gender'], y=mall_data['Spending Score'], order=list(sorted_mall_data.index))
    sns.jointplot(x=mall_data['Age'], y=mall_data['Spending Score'])
    sns.jointplot(x=mall_data['Annual Income'], y=mall_data['Spending Score'])
    plt.show()
    mall_data = mall_data[["Annual Income", "Spending Score"]]
    mall_data.to_csv(r".\\datasets\\mall_data.csv", index=False)


if __name__ == "__main__":
    visualise_dataset()