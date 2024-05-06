import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

iris_df = pd.read_csv(url, names=columns)

print(iris_df.head())


print("Feature Types:")
print(iris_df.dtypes)


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(12, 8))
for i, feature in enumerate(columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(iris_df[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
for i, feature in enumerate(columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='class', y=feature, data=iris_df)
    plt.title(f'Boxplot of {feature} by Class')
    plt.xlabel('Class')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
