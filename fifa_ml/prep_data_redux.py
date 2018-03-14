import numpy as np #lin alg
import pandas as pd #data processing

col_types = {'Overall': np.int32, 'Age': np.int32}
df = pd.read_csv('CompleteDataset.csv', usecols=['Name', 'Photo', 'Value', 'Overall', 'Age', 'Finishing'], dtype=col_types)

df['Value']=df['Value'].str.replace('\u20ac35', '')

def parseValue(strVal):
    if 'M' in strVal:
        return int(float(strVal.replace('M', ''))*1000000)
    elif 'K' in strVal:
        return int(float(strVal.replace('K',''))*1000)
    else:
        return int(strVal)

df['Value'] = df['Value'].apply(lambda x: parseValue(x))

df.isnull().sum()
