import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')


plt.figure(figsize=(8, 6))
sns.violinplot(x='fare', data=titanic, color='salmon')
plt.title('Distribution of Ticket Prices')
plt.xlabel('Fare')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(titanic['fare'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Ticket Prices')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

sns.pairplot(titanic, hue='survived', palette='husl', diag_kind='kde')
plt.show()
