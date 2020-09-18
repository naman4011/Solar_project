#Prediction


'''point = [float(x) for x in input('').split()]
point.append(1.367)

point = point*2
region = []
if point[0] >94 and point[1]>30 :
    region.append(1)
else:
    region.append(2)
'''
import numpy as np
import pandas as pd
import jan_model as jan
import feb_model as feb
import march_model as march
import april_model as april
import may_model as may
import june_model as june
import july_model as july
import aug_model as aug
import sept_model as sept
import oct_model as octo
import nov_model as nov
import dec_model as dec
import YearlyAvg
import newpoint

point = newpoint.point
test_arr = np.array(point[:4]).reshape(1,-1)
test_arr = test_arr*2

# Months Value Prediction

vals = []
models = [jan,feb,march,april,may,june,july,aug,sept,octo,nov,dec]
#model = jan
for model in models:
    regressor1 = model.regressor1
    regressor2 = model.regressor2
    regressor3 = model.regressor3
    regressor4 = model.regressor4
    regressor5 = model.regressor5
    arr = test_arr/2
    test = point[:4]
    val1 = regressor1.predict(arr)
    val3 = regressor3.predict(arr)
    test.append(val1[0,0])
    arr = np.array(test).reshape(1,-1)
    val2 = regressor2.predict(arr)
    test.append(val2[0,0]);test.append(val3[0,0])
    arr = np.array(test).reshape(1,-1)
    val4 = regressor4.predict(arr)
    test.append(val4[0,0])
    arr = np.array(test).reshape(1,-1)
    val5 = regressor5.predict(arr)
    v= [val1[0,0],val2[0,0],val3[0,0],val4[0,0],val5[0,0]]
    vals.append(v)

arr2 = np.array(vals)
data = {'MonthAvgI_HP':arr2[:,0],
        'MonthAvgI_IncP':arr2[:,1],
        'OptSlopeAngle':arr2[:,2],
        'MonthAvgI_B':arr2[:,3],
        'Ratio_D_G':arr2[:,4]}
df = pd.DataFrame(data,index = ['Jan','Feb','March','April','May','June','July','August','September','October','November','December'])

# Yearly Avg Prediction

arr = test_arr/2
test = point[:4]
regressor1 = YearlyAvg.regressor1
regressor2 = YearlyAvg.regressor2
regressor3 = YearlyAvg.regressor3
regressor4 = YearlyAvg.regressor4
regressor5 = YearlyAvg.regressor5
val1 = regressor1.predict(arr)
val3 = regressor3.predict(arr)
test.append(val1[0,0])
arr = np.array(test).reshape(1,-1)
val2 = regressor2.predict(arr)
test.append(val2[0,0]);test.append(val3[0,0])
arr = np.array(test).reshape(1,-1)
val4 = regressor4.predict(arr)
test.append(val4[0,0])
arr = np.array(test).reshape(1,-1)
val5 = regressor5.predict(arr)

YearlyAvg = [val1[0,0],val2[0,0],val3[0,0],val4[0,0],val5[0,0]]


'''#plot
months = ['jan','feb','march','april', 'may' , 'june','july','august', 'Sept','Oct','Nov','Dec'] 
plt.plot(months, df.iloc[:,0].values) 
plt.xlabel('Months') 
plt.ylabel('YearAvgI_HP')  
plt.title('YearAvgI_HP vs Months')
plt.show()'''


