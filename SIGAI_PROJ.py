import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Task 2: Load the dataset
dataset = pd.read_csv('Heart_disease_statlog.csv')  # Replace  with the actual filename

# Task 3: Analyze the dataset
print(dataset.head())  # Display the first few rows
print(dataset.describe())  # Display statistical analysis of the dataset

# Task 4: Plotting
# age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target are the names of columns to be plotted
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(dataset['age'])
plt.title('age')
plt.subplot(2, 2, 2)
plt.plot(dataset['sex'])
plt.title('sex')
plt.subplot(2, 2, 3)
plt.plot(dataset['cp'])
plt.title('cp')
plt.subplot(2, 2, 4)
plt.plot(dataset['trestbps'])
plt.title('trestbps')
plt.tight_layout()
plt.show()
plt.plot(dataset['chol'])
plt.title('chol')
plt.tight_layout()
plt.show()
plt.plot(dataset['fbs'])
plt.title('fbs')
plt.tight_layout()
plt.show()
plt.plot(dataset['restecg'])
plt.title('restecg')
plt.tight_layout()
plt.show()
plt.plot(dataset['thalach'])
plt.title('thalach')
plt.tight_layout()
plt.show()
plt.plot(dataset['exang'])
plt.title('exang')
plt.tight_layout()
plt.show()
plt.plot(dataset['oldpeak'])
plt.title('oldpeak')
plt.tight_layout()
plt.show()
plt.plot(dataset['slope'])
plt.title('slope')
plt.tight_layout()
plt.show()
plt.plot(dataset['ca'])
plt.title('ca')
plt.tight_layout()
plt.show()
plt.plot(dataset['thal'])
plt.title('thal')
plt.tight_layout()
plt.show()
plt.plot(dataset['target'])
plt.title('target')
plt.tight_layout()
plt.show()


# Task 5: Removing rows with null values
dataset.dropna(inplace=True)  # Remove rows with null values

# Task 6: Apply normalization or standardization
# Assuming you want to use StandardScaler for standardization
scaler = StandardScaler()
# Alternatively, you can use MinMaxScaler for normalization
# scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset)

# Task 7: Display correlation matrix
correlation_matrix = dataset.corr()
print(correlation_matrix)
