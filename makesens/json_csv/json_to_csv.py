
import pandas as pd
import numpy as np
import os

from pandas.io.json import json_normalize
import json



def toNumeric (data):
    for i in data.columns:
        data[i] = pd.to_numeric(data[i])
    return data



def JsonCsv (name):
    contenido = os.listdir(name)
    index = []
    result = pd.DataFrame()
    for i in range(0,len(contenido)):
        f =open(name + str(contenido[i]))
        json_file = json.load(f)
        a = pd.json_normalize(json_file['data'])
        
        col = []
        if 'sensor' in a.columns:
            for j in range(0,len(a['data_type'])):
                col.append(a['data_type'][j] + '_' + str(a['sensor'][j]))
            data =pd.DataFrame(columns =col)
            data.loc[0] = list(a['value'])
            index.append(json_file['timestamp'])
        else:
            continue
        
    
        if i == 0:
            result = data
        else:
            result = pd.concat([result,data],axis= 0)
    result.index = index

    return result




def json_to_csv_eva (name,resample = '1T'):
    carpetas = os.listdir(name)
    vars = ['var' + i for i in carpetas]

    for i in range(0,len(carpetas)):
        vars[i] = JsonCsv(name+carpetas[i]+'/')

    all_data = pd.concat(vars,axis=0)
    all_data.index = pd.DatetimeIndex(all_data.index).strftime('%Y-%m-%d %H:%M:%S')
    all_data.index = all_data.index.astype("datetime64[ns]")
    all_data = toNumeric(all_data)
    all_data = all_data.resample(resample).mean()
    
    return all_data



