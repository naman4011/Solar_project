# Solar Project

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Y_Solar_data.csv')
df = dataset.iloc[:,:].values
#df = dataset2.iloc[0:202,0:9]
#Studying the dataset
dataset['YearAvgI_HP'].describe()
dataset['YearAvgI_IncP'].describe()
dataset['YearAvgI_B'].describe()
dataset['OptSlopeAngle'].describe()
dataset['Ratio_D_G'].describe()

#correlation matrix
corrmat = dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,fmt='.2f');

#scatterplot
sns.set()
sns.pairplot(dataset, size = 2.5)
plt.show();

plt.scatter(df[:,0:1],df[:,1:2],color = 'green')
plt.ylim(7,39)
plt.xlim(67,98)
plt.show()