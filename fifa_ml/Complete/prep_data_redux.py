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
