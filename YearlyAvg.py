# Model

#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Y_Solar_data.csv')
X = dataset.iloc[:,:4].values
y = dataset.iloc[:,4:].values

#Dividing the dataset into features and target values
#Choosing the YearAvgI_HP as the target value and Longitudes ,
#latitudes ,elevation and the  ExtSR as the features
X1 = dataset.iloc[:,:4].values
y1 = y[:,0:1]

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train1,X_test1,y_train1,y_test1 = train_test_split(X1,y1,test_size = 0.3,random_state = 0)

#fitting simple linear regression model to Training set
regressor1 = LinearRegression()
regressor1.fit(X_train1 , y_train1)

y_test_predict1 = regressor1.predict(X_test1)


#Splitting the dataset for predecting the next feature YearAvgI_IncP
#Now for finding the YearAvgI_Inc, we consider the predicted value
#of YearAvgI_HP as also a feature as it shows a high correlation

X2 = dataset.iloc[:,0:5].values
y2 = y[:,1:2]

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train2,X_test2,y_train2,y_test2 = train_test_split(X2,y2,test_size = 0.3,random_state = 0)

#fitting simple linear regression model to Training set
regressor2 = LinearRegression()
regressor2.fit(X_train2 , y_train2)

y_test_predict2 = regressor2.predict(X_test2)

# Predicting OptSlopeAngle
# Using the Features Longitude,latitude,elevation ExtSR for predicting the OptSlopeAngle
X_train3,X_test3,y_train3,y_test3 = train_test_split(X,y[:,2:3],test_size = 0.3,random_state = 0)
regressor3 = LinearRegression()
regressor3.fit(X_train3,y_train3)

y_test_predict3 = regressor3.predict(X_test3)


#Predicting the YearAvgI_B
#Input Feature values - Latitude, Longitude, Elevation, ExtSR,YearAvgI_HP,YearAvgI_IncP,OptSlopeAngle

X4 = dataset.iloc[:,:7].values
y4 = y[:,3:4]
X_train4,X_test4,y_train4,y_test4 = train_test_split(X4,y4,test_size = 0.3,random_state = 0)

regressor4 = LinearRegression()
regressor4.fit(X_train4,y_train4)

y_test_predict4 = regressor4.predict(X_test4)


#Predecting the val of Ratio_D_G
#Input Features - All Remaining Features
X5 = dataset.iloc[:,:8].values
y5 = y[:,4:5]

X_train5,X_test5,y_train5,y_test5 = train_test_split(X5,y5,test_size = 0.3,random_state = 0)

regressor5 = LinearRegression()
regressor5.fit(X_train5,y_train5)

y_test_predict5 = regressor5.predict(X_test5)


'''# Plot

plt.scatter(X_train1[:,0:1] , y_train1 , color = 'red')
plt.scatter(X_train1[:,0:1] , regressor1.predict(X_train1) , color = 'blue')
plt.title('longitude VS YearAvgI_HP(training)')
plt.xlabel('Longitude')
plt.ylabel('YearAvgI_HP')
plt.show()

plt.scatter(X_test1[:,0:1] , y_test1 , color = 'red')
plt.scatter(X_test1[:,0:1] , regressor1.predict(X_test1) , color = 'blue')
plt.title('longitude VS YearAvgI_HP(test)')
plt.xlabel('Longitude')
plt.ylabel('YearAvgI_HP')
plt.show()


plt.scatter(X_train2[:,0:1] , y_train2 , color = 'red')
plt.scatter(X_train2[:,0:1] , regressor2.predict(X_train2) , color = 'blue')
plt.title('longitude VS YearAvgI_IncP(training)')
plt.xlabel('Longitude')
plt.ylabel('YearAvgI_IncP')
plt.show()

plt.scatter(X_test2[:,0:1] , y_test2 , color = 'red')
plt.scatter(X_test2[:,0:1] , regressor2.predict(X_test2) , color = 'blue')
plt.title('longitude VS YearAvgI_IncP(test)')
plt.xlabel('Longitude')
plt.ylabel('YearAvgI_IncP')
plt.show()

plt.scatter(X_train3[:,0:1] , y_train3 , color = 'red')
plt.scatter(X_train3[:,0:1] , regressor3.predict(X_train3) , color = 'blue')
plt.title('longitude VS OptAngle(training)')
plt.xlabel('Longitude')
plt.ylabel('OptAngle')
plt.show()

plt.scatter(X_test3[:,0:1] , y_test3 , color = 'red')
plt.scatter(X_test3[:,0:1] , regressor3.predict(X_test3) , color = 'blue')
plt.title('longitude VS OptSlopeAngle(test)')
plt.xlabel('Longitude')
plt.ylabel('OptSopeAngle')
plt.show()

plt.scatter(X_train4[:,0:1] , y_train4 , color = 'red')
plt.scatter(X_train4[:,0:1] , regressor4.predict(X_train4) , color = 'blue')
plt.title('longitude Vs YearAvgI_B(training)')
plt.xlabel('Longitude')
plt.ylabel('YearAvgI_B')
plt.show()

plt.scatter(X_test4[:,0:1] , y_test4 , color = 'red')
plt.scatter(X_test4[:,0:1] , regressor4.predict(X_test4) , color = 'blue')
plt.title('longitude VS YearAvgI_B(test)')
plt.xlabel('Longitude')
plt.ylabel('YearAvgI_B')
plt.show()

plt.scatter(X_train5[:,0:1] , y_train5 , color = 'red')
plt.scatter(X_train5[:,0:1] , regressor5.predict(X_train5) , color = 'blue')
plt.title('longitude Vs Ratio_D_G')
plt.xlabel('Longitude')
plt.ylabel('Ratio_D_G')
plt.show()

plt.scatter(X_test5[:,0:1] , y_test5 , color = 'red')
plt.scatter(X_test5[:,0:1] , regressor5.predict(X_test5) , color = 'blue')
plt.title('longitude VS Ratio_D_G')
plt.xlabel('Longitude')
plt.ylabel('Ratio_D_G')
plt.show()
'''


