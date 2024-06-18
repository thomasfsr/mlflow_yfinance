# MLflow to time series data

## Overview
MLflow is a well-established tool for monitoring ML models.  
This project aims to provide a hands-on application of this tool to monitor the pipeline of a model that will predict the price of any stock available in the YFinance library.  
In this case, I used the "close" price and volume of the asset for simplicity.   

## Transformation  
I created lag columns of the close price following the parameters of the class I created to prepare the dataset for use. The default number of lags was set to 30, resulting in the dataset being organized into sequences of 30 consecutive close prices and their respective volumes:
```python   
def lag_data(self, df:pd.DataFrame=None, lags:int=30, column:str = 'Close'):
        values = []
        volume = []
        for i in range(0,len(df)-lags,lags):
            v = df['Volume'].values[i:i+lags]
            d = df[column].values[i:i+lags]
            volume.append(v)
            values.append(d)
        val_array = np.array(values)
        vol_array = np.array(volume)
        val_arrayT = val_array.T
        vol_arrayT = vol_array.T
        val_dict = {f"t{i}": val_arrayT[i] for i in range(0, lags)}
        vol_dict = {f"t_vol{i}": vol_arrayT[i] for i in range(0, lags)}
        val_df = pd.DataFrame(val_dict)
        vol_df = pd.DataFrame(vol_dict)

        return val_df, vol_df
```
The number of days ahead to be predicted is also tuneable and the default is 5 days out of those 30 lag days.  
For instance:  
|      |   t0   |   t1    |   t2    |   t3    |   t4    |   t5    | ... |   t24   |   t25   |   t26   |   t27   |   t28   |   t29   |
|------|--------|---------|---------|---------|---------|---------|-----|---------|---------|---------|---------|---------|---------|
|  0   | 0.8225 | 0.82325 | 0.8265  | 0.82925 | 0.8285  | 0.842   | ... | 0.709   | 0.70125 | 0.73225 | 0.7325  | 0.7015  | 0.705   |
|  1   | 0.70525| 0.66075 | 0.6305  | 0.63725 | 0.63575 | 0.6325  | ... | 0.792   | 0.8055  | 0.8075  | 0.8035  | 0.8275  | 0.8205  |
|  2   | 0.84525| 0.84775 | 0.84625 | 0.86075 | 0.862   | 0.87075 | ... | 0.90675 | 0.91125 | 0.912   | 0.93275 | 0.905   | 0.88825 |
|  3   | 0.9    | 0.88925 | 0.869   | 0.87425 | 0.88325 | 0.882   | ... | 1.156   | 1.15825 | 1.1545  | 1.1845  | 1.155   | 1.16975 |
|  4   | 1.172  | 1.184   | 1.18875 | 1.168   | 1.189   | 1.18175 | ... | 1.338   | 1.3555  | 1.3305  | 1.36675 | 1.392   | 1.41575 |



