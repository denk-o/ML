# coding=utf-8
import numpy as np #lin alg
import pandas as pd #data processing

col_types = {'Overall': np.int32, 'Age': np.int32}
df = pd.read_csv('../Complete/CompleteDataset.csv', usecols=['Name', 'Photo', 'Value', 'Overall', 'Age', 'Finishing'], dtype=col_types)

df['Value']=df['Value'].str.replace('€', '')

def parseValue(strVal):
    if 'M' in strVal:
        return int(float(strVal.replace('M', ''))*1000000)
    elif 'K' in strVal:
        return int(float(strVal.replace('K',''))*1000)
    else:
        return int(strVal)

df['Value'] = df['Value'].apply(lambda x: parseValue(x))

print(df.isnull().sum())

#remove bad values, anomalies/out of bounds
df = df.loc[df.Value>0]
def between_1_and_99(s):
    try:
        n=int(s)
        return(1<=n and n<=99)
    except ValueError:
        return False

df = df.loc[df['Finishing'].apply(lambda x: between_1_and_99(x))]

df['Finishing']=df['Finishing'].astype('int')

print(df.head())
print('Describe: ')
print(df.describe())


#part 2
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.20, random_state=99)
xtrain = train[['Value']]
ytrain = train[['Overall']]

xtest = test[['Value']]
ytest = test[['Overall']]

#create linear regression object

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(xtrain, ytrain)
y_pred = regr.predict(xtest)

import matplotlib.pyplot as plt

plt.scatter(xtest, ytest, color ='black')
plt.plot(xtest, y_pred, color='blue', linewidth=3)
plt.xlabel("Value")
plt.ylabel("overall")
#plt.show()

from sklearn.metrics import mean_squared_error, r2_score

print("Mean squared error: %.2f" % r2_score(ytest,y_pred))
print('Variance score: %.2f' % r2_score(ytest, y_pred))

