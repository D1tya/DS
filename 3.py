import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['Category'] = iris.target_names[iris.target]


summary_stats = df.groupby('Category')['sepal length (cm)'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()

print(summary_stats)
import pandas as pd

iris_df = pd.read_csv('iris.csv')

setosa_stats = iris_df[iris_df['species'] == 'Iris-setosa'].describe()
versicolor_stats = iris_df[iris_df['species'] == 'Iris-versicolor'].describe()
virginica_stats = iris_df[iris_df['species'] == 'Iris-virginica'].describe()

print("Statistical Details for Iris-setosa:")
print(setosa_stats)

print("\nStatistical Details for Iris-versicolor:")
print(versicolor_stats)

print("\nStatistical Details for Iris-virginica:")
print(virginica_stats)
