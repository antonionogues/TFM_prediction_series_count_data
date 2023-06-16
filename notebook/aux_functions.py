import pandas as pd
import numpy as np
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError


# Detectar outliers

def detectar_outliers(dataframe, columna, umbral=1.5):
    # Calcular los cuartiles
    q1 = dataframe[columna].quantile(0.25)
    q3 = dataframe[columna].quantile(0.75)

    # Calcular el rango intercuartil (IQR)
    iqr = q3 - q1

    # Calcular los límites superior e inferior
    limite_inferior = q1 - umbral * iqr
    limite_superior = q3 + umbral * iqr

    # Detectar outliers
    outliers = dataframe[(dataframe[columna] < limite_inferior) | (dataframe[columna] > limite_superior)]

    return outliers


# Transformar dataframe de precipitaciones a formato correcto

def transformar_dataframe(df):
    # Cambiar los valores 'Ip' a 0
    df['prec'] = df['prec'].replace('Ip', 0)
    
    # Cambiar las comas por puntos
    df['prec'] = df['prec'].str.replace(',', '.')
    
    # Convertir las columnas a formato fecha y float
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['prec'] = df['prec'].astype(float)
    
    return df

# Crear la columna booleana para contar las precipitaciones 

def crear_columna_booleana(df):
    df['precipitaciones'] = df['prec'].apply(lambda x: 1 if x != 0 else 0)
    return df

# Cuenta las precipitaciones por mes 

def contar_precipitaciones_por_mes(df):  
    df['mes'] = df['fecha'].dt.to_period('M')
    df_precipitaciones = df.groupby('mes')['precipitaciones'].sum().reset_index()
    df_precipitaciones.columns = ['fecha', 'prec']
    
    return df_precipitaciones

# MAPE Y RMSE

RMSE = MeanSquaredError(square_root=True)
MAPE = MeanAbsolutePercentageError(symmetric=False)

def ForecastPerformance(original, forecast):
    rmse_value = round(RMSE(original, forecast), 2)
    mape_value = round(MAPE(original, forecast) * 100, 2)
    results = {'RMSE': rmse_value, 'MAPE': mape_value}
    df_results = pd.DataFrame(results, index=[0])
    return df_results

