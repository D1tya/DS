import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Student_ID': [1, 2, 3, 4, 5],
    'Math_Score': [85, 90, 75, np.nan, 95],
    'Science_Score': [80, 85, 70, 65, 90],
    'English_Score': [70, 75, 60, 55, 80],
    'Attendance_Rate': [0.95, 0.90, 0.85, 0.75, 0.98]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

df.fillna(df.mean(), inplace=True)

numeric_cols = ['Math_Score', 'Science_Score', 'English_Score', 'Attendance_Rate']
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot of Numeric Variables")
plt.xticks(rotation=45)
plt.show()


df['Log_Attendance_Rate'] = np.log(df['Attendance_Rate'])

print("\nDataset after transformations:")
print(df)
