import yfinance as yf
from datetime import datetime
import pandas as pd
import os
from typing import Literal
import numpy as np

class GetData:
    def __init__(self, ticker_symbol:str='BTC-USD', start:str='2015-12-20',end:str= datetime.now().strftime('%Y-%m-%d')):
        self.ticker_symbol = ticker_symbol
        self.start = start
        self.end = end

    def get_data_df(self):
        try:
            data = yf.download( self.ticker_symbol, start=self.start, end=self.end)
            df = pd.DataFrame(data)
        except Exception as e:
            print(f"Error downloading data: {e}")
            df = pd.DataFrame()  # Return an empty DataFrame in case of error
        return df
    
    def export(self, df:pd.DataFrame, file_type: Literal['parquet', 'csv', 'json']):
        name = self.ticker_symbol.split('-')[0]
        os.makedirs('data', exist_ok=True)
        if file_type == 'parquet':
            df.to_parquet(f'data/{name}.parquet')
        if file_type == 'csv':
            df.to_csv(f'data/{name}.csv')
        if file_type == 'json':
            df.to_json(f'data/{name}.json')
    
    def smooth_data(self, df:pd.DataFrame=None, alpha:float=0.2):
        smooth_df = df.apply(lambda column: column.ewm(alpha=alpha, adjust=False).mean())
        return smooth_df
    
    def lag_data(self, df:pd.DataFrame=None, lags:int=30, column:str = 'Close'):
        values = []
        volume = []
        for i in range(0,len(df)-lags,lags):
            v = sum(df['Volume'].values[i:i+lags])
            d = df[column].values[i:i+lags]
            volume.append(v)
            values.append(d)
        varray = np.array(values)
        varrayT = varray.T
        vdict = {f"t{i}": varrayT[i] for i in range(0, lags)}
        vdict['vol']= volume
        df_lag = pd.DataFrame(vdict)
        return df_lag
    
    def sum_vol(self,df:pd.DataFrame):
        return sum(df['Volume'][:30])
        
    def func_start(self):
        df = self.get_data_df()
        df_lag = self.lag_data(df,30,'Close')
        return df_lag