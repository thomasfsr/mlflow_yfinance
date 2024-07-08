from src.finance_get import GetData
from src.model_ensemble import GbmModel
from sklearn.model_selection import ParameterGrid
import yaml
import mlflow

with open('src/config.yml', 'r') as f:
    config = yaml.safe_load(f)

param_grid = {
    'model': config['parameters']['model'],
    'split_type': config['parameters']['split_type'],
    'n_estimators': config['parameters']['n_estimators'],
    'lr': config['parameters']['learning_rate'],
    'steps': config['parameters']['steps'],
    'max_depth': config['parameters']['max_depth'],
    'reg_alpha' : config['parameters']['reg_alpha'],
    'reg_lambda': config['parameters']['reg_lambda']
    }
grid = ParameterGrid(param_grid=param_grid)

# Start Experiment:
mlflow.create_experiment("Gold1")
mlflow.set_experiment("Gold1")
get_obj = GetData(ticker_symbol='CL=F')

for params in grid:
    steps = params['steps']
    val_df, vol_df = get_obj.val_vol_datasets(lags=30,step=steps)
    model = GbmModel(val=val_df, vol= vol_df,n_X=25)
    model.model(
        model = params['model'],
        n_estimators=params['n_estimators'], 
        lr=params['lr'], 
        split_type= params['split_type'],
        max_depth = params['max_depth'],
        reg_alpha = params['reg_alpha'],
        reg_lambda = params['reg_lambda'],
        steps=params['steps']
        )