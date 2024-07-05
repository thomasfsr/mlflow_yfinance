from src.finance_get import GetData
from src.model_ensemble import GbmModel
from sklearn.model_selection import ParameterGrid
import yaml
import mlflow

with open('src/config.yml', 'r') as f:
    config = yaml.safe_load(f)

param_grid = {
    'split_type': config['parameters']['split_type'],
    'n_estimators': config['parameters']['n_estimators'],
    'lr': config['parameters']['learning_rate'],
    'steps': config['parameters']['steps']
}
grid = ParameterGrid(param_grid=param_grid)

# Start Experiment:
mlflow.set_experiment("Gold")
get_obj = GetData(ticker_symbol='CL=F')

for params in grid:
    steps = params['steps']
    val_df, vol_df = get_obj.val_vol_datasets(lags=30,step=steps)
    mlflow.log_param('steps',steps)
    model = GbmModel(val=val_df, vol= vol_df,n_X=25)
    model.model(n_estimators=params['n_estimators'], 
                lr=params['lr'], 
                split_type= params['split_type']
                )