# Trail Model on Monthly Dataset

# Importing the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#r = int(input('which region does this belong'))
 
# Importing the dataset
#month_dataset = pd.read_csv('M_Solar_data.csv')
#df = pd.read_csv('M_Solar_data.csv') 
import Clusters
from newpoint import region

if region[0]==2:
    df = Clusters.dataframe2
if region[0]==1:
    df = Clusters.dataframe1
columns = list(df.columns)
for i in columns[1:]:
    df[i] = df[i].astype('float64')

# Label Encoding
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
df['Month'] = labelencoder.fit_transform(df['Month'])
df2 = df.iloc[:,:].values
jan_data = [];feb_data=[];march_data=[];april_data = [];may_data=[];june_data=[]
july_data =[];aug_data=[];sept_data=[];oct_data=[];nov_data=[];dec_data=[]

for i in range(len(df2[:,0])):
    if df2[i,0]==0:
        jan_data.append(list(df2[i,1:]))
    if df2[i,0]==1:
        feb_data.append(list(df2[i,1:]))
    if df2[i,0]==2:
        march_data.append(list(df2[i,1:]))
    if df2[i,0]==3:
        april_data.append(list(df2[i,1:]))
    if df2[i,0]==4:
        may_data.append(list(df2[i,1:]))
    if df2[i,0]==5:
        june_data.append(list(df2[i,1:]))
    if df2[i,0]==6:
        july_data.append(list(df2[i,1:]))
    if df2[i,0]==7:
        aug_data.append(list(df2[i,1:]))
    if df2[i,0]==8:
        sept_data.append(list(df2[i,1:]))
    if df2[i,0]==9:
        oct_data.append(list(df2[i,1:]))
    if df2[i,0]==10:
        nov_data.append(list(df2[i,1:]))
    if df2[i,0]==11:
        dec_data.append(list(df2[i,1:]))

jan_data = np.array(jan_data)
feb_data = np.array(feb_data)
march_data = np.array(march_data)
april_data = np.array(april_data)
may_data = np.array(may_data)
june_data = np.array(june_data)
july_data = np.array(july_data)
aug_data = np.array(aug_data)
sept_data = np.array(sept_data)
oct_data = np.array(oct_data)
nov_data = np.array(nov_data)
dec_data = np.array(dec_data)
data = [jan_data,feb_data,march_data,april_data,
        may_data,june_data,july_data,aug_data,sept_data,oct_data,nov_data,dec_data]

'''#Plot of mean values
mean_vals = []
for k in data:
    ele = np.mean(k[:,])
    mean_vals.append(ele)
months = ['jan','feb','march','april', 'may' , 'june','july','august', 'Sept','Oct','Nov','Dec'] 
plt.plot(months, mean_vals) 
plt.xlabel('Months') 
plt.ylabel('YearAvgI_HP')  
plt.title('YearAvgI_HP vs Months')
plt.show()
'''
