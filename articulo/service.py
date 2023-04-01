import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def nrmse(dat1, dat2):
    return np.sqrt(np.mean(np.log10((dat1+1)/(dat2+1))**2))

def cutdata(data, start_date:str, end_date:str):
    """
    Función que selecciona un subconjunto de datos en un rango de fechas específico.

    Args:
        data (pd.DataFrame): DataFrame que contiene los datos a filtrar.
        start_date (str): Fecha de inicio del rango en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de finalización del rango en formato 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: Nuevo DataFrame con los datos filtrados en el rango de fechas especificado.

    Ejemplo:
        data_filtered = cutdata(data, '2021-01-01', '2021-01-31')
    """

    # Crea una máscara booleana para seleccionar las filas dentro del rango de fechas
    mask = (data.index >= start_date) & (data.index <= end_date)

    # Aplica la máscara al DataFrame original para obtener el subconjunto de datos
    data = data[mask]

    return data

def load_racimo_data(file_path:str):
    """
    Carga y procesa los datos de calidad del aire de Racimo desde un archivo CSV.
    
    Parameters
    ----------
    file_path : str
        La ruta al archivo CSV que contiene los datos de calidad del aire de Racimo.
        
    Returns
    -------
    pivot_data : pd.DataFrame
        Un DataFrame que contiene los datos de calidad del aire procesados.
    """
    # Leer datos del archivo CSV
    data = pd.read_csv(file_path, delimiter=',')
    
    # Eliminar filas con 'alert' como id_parametro
    data = data.drop(data[data.id_parametro == 'alert'].index)
    
    # Convertir la columna 'valor' a numérico
    data.valor = pd.to_numeric(data.valor)
    
    # Pivoteo de los datos
    pivot_data = pd.pivot_table(data, index='fecha_hora_med', columns='id_parametro', 
                                 values='valor').reset_index().set_index("fecha_hora_med")
    
    # Conservar solo las columnas necesarias y las variables importantes ("pm10_a", "pm25_a", "t", "p", "h")
    cols_keep = ["pm10", "pm10_a", "pm25", "pm25_a", "t", "p", "h"]
    pivot_data = pivot_data[cols_keep]
    
    # Ajustar la zona horaria
    pivot_data.index = pd.DatetimeIndex(pivot_data.index) - pd.Timedelta(hours=5)
    
    # Remuestrear los datos con promedios por hora
    pivot_data = pivot_data.resample('60T').mean()
    
    # Dar formato al índice como una cadena
    pivot_data.index = pivot_data.index.strftime('%Y-%m-%d %H:%M:%S')
    
    # Eliminar nombres de índice y columna
    pivot_data.index.name = None
    pivot_data.columns.name = None
    
    # Convertir el índice de nuevo a datetime
    pivot_data.index = pd.to_datetime(pivot_data.index)
    
    return pivot_data

def clean_nan(data, var):
    """
    Limpia los valores no numéricos o inválidos en la columna especificada de un DataFrame de pandas.
    
    Args:
        data (pd.DataFrame): DataFrame que contiene la columna a limpiar.
        var (str): Nombre de la columna a limpiar.
    
    Returns:
        pd.DataFrame: DataFrame con la columna especificada limpiada.
    """
    b = np.array(data[var])
    b = np.where(b == 'NoData', np.nan, b)
    b = np.where(b == '---', np.nan, b)
    b = np.where(b == '<Samp', np.nan, b)
    b = np.where(b == 'OffScan', np.nan, b)
    b = np.where(b == 'Zero', np.nan, b)
    b = np.where(b == 'Calib', np.nan, b)
    b = np.where(b == 'InVld', np.nan, b)
    data[var] = list(b)
    return data


def load_amb_data(file_path:str):
    """
    Carga los datos de un archivo Excel de datos AMB y limpia los valores no numéricos o inválidos.
    
    Args:
        name (str): Ruta y nombre del archivo Excel (.xlsx) a cargar.
    
    Returns:
        pd.DataFrame: DataFrame con los datos AMB cargados y limpios.
    """
    data_AMB = pd.read_excel(file_path, header=0)
    # data_AMB.columns = list(data_AMB.iloc[0])
    data_AMB.index = list(data_AMB["Date&Time"])
    data_AMB = data_AMB[3:-8]
    data_AMB.index = pd.DatetimeIndex(data_AMB.index, dayfirst=True)
    data_AMB.index = data_AMB.index.astype(str)
    data_AMB = data_AMB.drop(['Date&Time'], axis=1)
    data_AMB.index = pd.to_datetime(data_AMB.index)

    for i in data_AMB.columns:
        data_AMB = clean_nan(data_AMB, i)
    for i in data_AMB.columns:
        data_AMB[i] = pd.to_numeric(data_AMB[i])
    return data_AMB


def linear_calibration_model(data_x, data_y):
    """
    Fits a linear calibration model using the given data_x and data_y.
    
    Parameters
    ----------
    data_x : array-like or pd.Series
        The independent variable data.
    data_y : array-like or pd.Series
        The dependent variable data.
        
    Returns
    -------
    results : dict
        A dictionary containing the following performance metrics:
            - RMSE (Root Mean Squared Error)
            - COEF (Coefficient)
            - Intercept
            - r2_score (R-squared)
            - correlation
            - NRMSE (Normalized Root Mean Squared Error)
    """
    
    def nrmse(dat1, dat2):
        return np.sqrt(np.mean(np.log10((dat1 + 1) / (dat2 + 1)) ** 2))

    X = pd.DataFrame(data_x)
    Y = data_y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=9)
    lin_reg_mod = LinearRegression()
    lin_reg_mod.fit(X_train, y_train)
    pred = lin_reg_mod.predict(X_test)
    test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

    results = {
        'RMSE': test_set_rmse,
        'COEF': lin_reg_mod.coef_,
        'Intercept': lin_reg_mod.intercept_,
        'r2_score': r2_score(lin_reg_mod.predict(X), Y),
        'correlation': np.corrcoef(lin_reg_mod.predict(X), Y)[0][1],
        'NRMSE': nrmse(y_test, pred)
    }

    return results

def random_forest_calibration(data_x, data_y, n_estimators=800, test_size=0.5, random_seed=None):
    """
    Esta función entrena un modelo de Random Forest con los datos proporcionados y devuelve métricas de evaluación.
    
    Args:
    data_x (DataFrame): Un DataFrame que contiene las variables independientes (predictores).
    data_y (Series): Una Serie que contiene la variable dependiente (objetivo).
    n_estimators (int, opcional): El número de árboles en el bosque. Por defecto es 800.
    test_size (float, opcional): La proporción del conjunto de datos a incluir en la división de prueba. Por defecto es 0.5.
    random_seed (int, opcional): Una semilla para el generador de números aleatorios. Si no se proporciona, se generará una aleatoriamente.
    
    Returns:
    results (dict): Un diccionario que contiene las métricas de evaluación del modelo entrenado.
    """
    if random_seed is None:
        random_seed = np.random.randint(700)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size, random_state=random_seed)
    
    # Entrenar un modelo de Random Forest con los datos de entrenamiento
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_seed)
    rf.fit(X_train, y_train)
    
    # Predecir los valores objetivo para el conjunto de prueba
    predictions = rf.predict(X_test)
    
    # Calcular el error absoluto y las métricas de evaluación
    errors = abs(predictions - y_test)
    mape = 100 * abs(errors / y_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    accuracy = 100 - np.mean(mape)

    # Almacenar las métricas de evaluación en un diccionario
    results = {
        'RMSE': rmse,
        'r2_score': r2_score(predictions, y_test),
        'correlation': np.corrcoef(predictions, y_test)[0][1],
        'NRMSE': nrmse(y_test, predictions),
        'accuracy': accuracy
    }

    return rf.predict(data_x) ,results





