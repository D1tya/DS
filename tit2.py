import seaborn as sns
import matplotlib.pyplot as plt


titanic = sns.load_dataset('titanic')

# Plot box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic)
plt.title('Age Distribution by Gender and Survival')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()
