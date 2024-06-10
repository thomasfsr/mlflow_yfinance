from finance_get import GetData
from model_ensemble import GbmModel
import yaml
import mlflow

with open('src/config.yml', 'r') as f:
    config = yaml.safe_load(f)

mlflow.set_experiment("Yfinance Time Series - NVDA")
get_obj = GetData(ticker_symbol='NVDA')
df = get_obj.func_start()

model = GbmModel(df=df)

X_train, X_test, y_train, y_test = model.splitting()

for lr in config['parameters']['learning_rate']:
    for n_e in config['parameters']['n_estimators']:
        model.model(X_train=X_train, 
                    y_train= y_train, 
                    n_estimators= n_e,
                    lr= lr)
