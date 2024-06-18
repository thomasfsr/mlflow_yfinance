# MLflow to time series data

## Overview
MLflow is a well-estabilish tool to monitor ML models.
This project aims to get a hands-on application of this tool to monitor the pipeline of a model that will be used to predict the price of any stock market available on the library YFinance.  
In this case I used the "close" price and the volume of the asset for simplicity.

## Transformation  
I created lag columns of the close price following the parameters of the class I created to prepare the dataset to be used.  
The default number of lags was 30, so the dataset consists of slices of 30 sequences of close prices and its volume:  
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



