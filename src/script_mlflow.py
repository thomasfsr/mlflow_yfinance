from finance_get import GetData
from model_ensemble import GbmModel
from sklearn.model_selection import ParameterGrid
import yaml
import mlflow

with open('src/config.yml', 'r') as f:
    config = yaml.safe_load(f)

param_grid = {
    'split_type': config['parameters']['split_type'],
    'n_estimators': config['parameters']['n_estimators'],
    'lr': config['parameters']['learning_rate']
}
grid = ParameterGrid(param_grid=param_grid)

# Start Experiment:
mlflow.set_experiment("Yfinance Time Series - NVDA")
get_obj = GetData(ticker_symbol='NVDA')

# With sum of volumes as one of the features:
val_df, vol_df = get_obj.val_vol_datasets()
model = GbmModel(val=val_df, vol= vol_df)

for params in grid:
    model.model(n_estimators=params['n_estimators'], lr=params['lr'], split_type= params['split_type'])

# With the volumes of each timestamp:
val, vol = get_obj.val_vol_datasets()

model_2 = GbmModel(val=val, vol=vol)

for lr in config['parameters']['learning_rate']:
    for n_e in config['parameters']['n_estimators']:
        model_2.model(n_estimators= n_e,
                    lr= lr)