import pandas as pd
import numpy as np
import os



from pandas.io.json import json_normalize
import json
from datetime import datetime, timedelta

from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sfm 

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.externals import joblib 
from sklearn.preprocessing import LabelEncoder 
from numpy.core.umath_tests import inner1d


###################################  load data   ################################### 

def loaddata_racimo(name,station,window,resample = '1T'):
    """
    Parameters:
    * name     --> name of the file to load
    * station  --> number associated with the station
    * window   --> window width for smoothing
    * resample --> resampling

    Retunr:
    * DataFrame
    """
    data=pd.read_csv(name,delimiter=',')
    data=data.drop(data[data.id_parametro =='alert'].index)
    data.valor=pd.to_numeric(data.valor)
    pivot_data=pd.pivot_table(data,index='fecha_hora_med', columns='id_parametro', 
                              values='valor').reset_index().set_index("fecha_hora_med")    
    #keep really needed cols and important variables ("pm10_a","pm25_a","t","p","h")
    cols_keep=["pm10","pm10_a","pm25","pm25_a","t","p","h"]
    pivot_data=pivot_data[cols_keep]    
    pivot_data.index = pd.DatetimeIndex(pivot_data.index) - pd.Timedelta(hours = 5)
    pivot_data  = pivot_data.resample(resample).mean()
    pivot_data.index = pivot_data.index.strftime('%Y-%m-%d %H:%M:%S')
    for i in pivot_data.columns:
        pivot_data[i]=rolling(pivot_data[i],window)
    return pivot_data


def loaddata_amb(name,format):
    """
    Parameters:
    * name     --> name of the file to load

    Retunr:
    * DataFrame
    """
    if format == 'csv':
        data_AMB = pd.read_csv(name, header= 0)
    if format == 'xlsx':
        data_AMB = pd.read_excel(name, header= 0)
    
    data_AMB.columns = list(data_AMB.iloc[1])
    data_AMB.index = list(data_AMB["Date&Time"])
    data_AMB = data_AMB[3:-8] #removi la fila 1 (la de las unidades)
    data_AMB.index = pd.DatetimeIndex(data_AMB.index,dayfirst = True) 
    data_AMB.index = data_AMB.index.astype(str)
    data_AMB = data_AMB.drop(['Date&Time'], axis=1)
    for i in data_AMB.columns:
        data_AMB=clean_nan(data_AMB,i)
    return data_AMB

def loaddata_davis(name):
    Davis = pd.read_csv(name, sep="	", header=None)
    Davis.columns = ["{}_{}".format(Davis.iloc[0][i], Davis.iloc[1][i]).replace("nan_","") for i in Davis.columns]
    Davis = Davis.drop(index=[0,1])
    _Time=[]
    for i in Davis["Time"]:
        if(len(i) == 4): _Time.append("0"+i)
        else: _Time.append(i) 

    Davis["Date_Time"] = Davis["Date"] +"_" + _Time #String Date Time
    Davis["Date_Time"] = [datetime.strptime(i, "%d/%m/%y_%H:%M") + timedelta(days=33,hours=1+5,minutes=14) for i in Davis["Date_Time"]] #Lista de DateTime y correccion de tiempo
    Davis.drop(columns = ["Date", "Time"], inplace=True) #Elimina Columnas originales de Date y Time
    Davis.index = Davis["Date_Time"]
    Davis.drop(columns = ["Date_Time"], inplace = True) 
    keys_floats = list(Davis.columns[0:6].values) + list(Davis.columns[7:9]) + list(Davis.columns[10:-1].values)
    Davis[keys_floats] = Davis[keys_floats].astype("float")
    return Davis


def loaddata_eva(name):
    data = pd.read_csv(name, sep=',')
    data = data[['timestamp', 'sensor', 'pm10', 'pm2.5', 'pm1', 'noise', 'illuminance',
        'irradiance', 'temperature', 'humidity', 'presure']]
    data["timestamp"] = data["timestamp"].astype("datetime64")
    dat2 = data.drop(data[data.sensor != 2.0 ].index)
    dat1 = data.drop(data[data.sensor == 2.0 ].index)
    dat2.drop(columns = [ 'noise', 'illuminance', 'irradiance', 'temperature', 'humidity', 'presure'], inplace = True)

    result1 = pd.DataFrame(index=data['timestamp'])
    result2 = pd.DataFrame(index=data['timestamp'])
    for i in dat1.columns[2:]:
        var1 = Get_var(dat1,i)
        result1 = pd.merge(result1,var1,left_index=True,right_index=True)

    for i in dat2.columns[2:]:
        var2 = Get_var(dat2,i)
        result2 = pd.merge(result2,var2,left_index=True,right_index=True)

    result1.columns = ['pm10_2', 'pm2.5_2', 'pm1_2', 'noise', 'illuminance',
       'irradiance', 'temperature', 'humidity', 'presure']
    result2.columns = ['pm10_n', 'pm2.5_n', 'pm1_n']

    result = pd.DataFrame()
    result = pd.merge(result1,result2,left_index=True,right_index=True)
    result  =result.reindex(columns = ['pm10_2', 'pm2.5_2', 'pm1_2','pm10_n', 'pm2.5_n', 'pm1_n', 'noise', 'illuminance', 'irradiance', 'temperature', 'humidity', 'presure'])
    result = result.resample('1T').mean()
    result = result.drop_duplicates()
    return result

###########################################################################################################
def json_to_csv_eva(name):
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





##########################################################################################################

def Get_var(Data,name_variable):
    variable = json_normalize(  [json.loads(i) for i in Data[name_variable].dropna()] )
    variable["Date_Time"] = list(Data[["timestamp",name_variable]].dropna()["timestamp"])
    variable.index= variable['Date_Time']
    return variable['value']

def rolling(y,n):  #Rolling data to smooth values
    rolling = y.rolling(window = n , center = True , min_periods = 1)
    return  rolling.median()

def clean_nan(data,var):
    b = np.array(data[var])
    b = np.where(b == 'NoData',np.nan,b)
    b = np.where(b == '---',np.nan,b)
    data[var] = list(b)
    return data

def cutdata(datas:list,start_date,end_date):
    result = []
    for i in datas:
        mask = (i.index >= start_date) & (i.index <= end_date)
        i=i[mask]
        result.append(i)
    return result

def renamecol(datas:list,station:list):
    result = []
    for i in range(0,len(datas)):
        result.append(datas[i].rename(columns = {'pm10':'pm10_'+ str(station[i]),'pm10_a':'pm10_a'+str(station[i]), 'pm25':'pm25_'+str(station[i]),      'pm25_a':'pm25_a'+str(station[i]),'t': 't_'+str(station[i]), 'p': 'p_'+str(station[i]), 'h': 'h_'+str(station[i])}))
    return result

def  Merge(datas:list):
    result = pd.DataFrame()
    for i in range(0,len(datas)-1):
        if i == 0:
            result  = pd.merge(datas[i],datas[i+1],left_index=True,right_index=True)
        else:
            result  = pd.merge(result,datas[i+1],left_index=True,right_index=True)
    result = result.dropna()    
    return result

######################################################################################################################
def LinearModel(variables,porcentage):
    """
    Parameters:
        - Variables:list -->
        - Porcentage:float  --> 
    

    Returns:
        - RMSE:float --> mean square error
        - coefficients:list --> 
        - Intercept:float  -->  
    """

    Y = variables[0]
    X = pd.DataFrame({str(i):variables[i] for i in range(1,len(variables))})
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = porcentage, random_state=9)
    lin_reg_mod = LinearRegression()
    lin_reg_mod.fit(X_train, y_train)
    pred = lin_reg_mod.predict(X_test)
    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
    coef = lin_reg_mod.coef_
    intercept = lin_reg_mod.intercept_
    Yc = sum([variables[i] * coef[i-1] for i in range(1,len(variables))] ) + intercept
    return lin_reg_mod.coef_, lin_reg_mod.intercept_ , Yc
    #return Yc

def RamdonForest(variables,porcentage):
    """
    Parameters:
        - Variables:list -->
        - Porcentage:float  --> 
    

    Returns:
        - MAE:float  --> mean absolute error
        - RMSE:float --> mean square error
        - Accuracy:float --> 
    """

    Y = variables[0]
    X = pd.DataFrame({str(i):variables[i] for i in range(1,len(variables))})
    train_features,test_features,train_labels,test_labels=train_test_split(X,Y,test_size=porcentage,random_state=0)
    rf=RandomForestRegressor(n_estimators=800,random_state=0)
    rf.fit(train_features,train_labels)
    predictions=rf.predict(test_features)
    errors=abs(predictions-test_labels)
    mape=100*abs(errors/test_labels)
    rmse=np.sqrt(np.mean(errors**2))
    accuracy=100-np.mean(mape)
    mae = np.mean(errors)

    #return mae,rmse, accuracy, rf.predict(X)
    return pd.Series(rf.predict(X),variables[0].index)