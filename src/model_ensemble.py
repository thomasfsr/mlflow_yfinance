from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as mse
from xgboost import XGBRegressor as xgbr
from lightgbm import LGBMRegressor as lgbm
class GbmModel:
    def __init__(self,
                 val:pd.DataFrame = None,
                 vol:pd.DataFrame = None,
                 n_X:str = 15, 
                 splits:str = 5, 
                 rs:int = 42, 
                 ts:str= .3,
                 ):
        
        self.splits = splits
        self.n_X = n_X
        self.rs = rs
        self.ts = ts
        self.val = val
        self.vol = vol

    def splitting(self):
        if self.val is not None and self.vol is not None:
            n_X = self.n_X
            ts = self.ts
            rs = self.rs
            val = self.val.copy()
            vol = self.vol.copy()

            X_val = val.iloc[:, :n_X]
            X_vol = vol.iloc[:, :n_X]
            X = pd.concat([X_val, X_vol], axis=1)

            y = val.iloc[:, n_X:]
        else:
            ValueError()
        X_train, X_test, y_train, y_test = train_test_split(X, y,  
                                                    test_size=ts, 
                                                    shuffle=True, 
                                                    random_state=rs)

        return X_train, X_test, y_train, y_test

    def splitting_sum_vol(self):
        if self.val is not None and self.vol is not None:
            n_X = self.n_X
            ts = self.ts
            rs = self.rs
            val = self.val.copy()
            vol = self.vol.copy()

            X_val = val.iloc[:, :n_X]
            X_vol = vol.iloc[:, :n_X]
            X_vol = X_vol.sum(axis=1)
            X_vol = X_vol.to_frame('sum_vol')
            X = pd.concat([X_val, X_vol], axis=1)
            y = val.iloc[:, n_X:]
        else:
            ValueError()
        X_train, X_test, y_train, y_test = train_test_split(X, y,  
                                                    test_size=ts, 
                                                    shuffle=False,
                                                    )

        return X_train, X_test, y_train, y_test
    
    def split_load(self, split_type: str = 'sum_vols'):

        assert split_type in ['all_vols', 'sum_vols'], "split_type must be 'all_vols' or 'sum_vols'"

        function_mapping = {
            'all_vols': self.splitting,
            'sum_vols': self.splitting_sum_vol
        }

        split_function = function_mapping.get(split_type)

        return split_function()


    def model(self, 
              lr: float = .1, 
              n_estimators: int = 100,
              max_depth = 3,
              reg_alpha = None,
              reg_lambda = None,
              model:str = 'xb',
              steps:int = 10,
              split_type:str = 'sum_vols'
              ):
        
        X_train, X_test, y_train, y_test = self.split_load(split_type=split_type)

        n_splits = self.splits
        rs = self.rs

        xb = xgbr(n_estimators=n_estimators, 
                            random_state=rs, 
                            learning_rate=lr,
                            max_depth=max_depth,
                            reg_alpha = reg_alpha,
                            reg_lambda = reg_lambda
                            )
        lb = lgbm(n_estimators=n_estimators, 
                            random_state=rs, 
                            learning_rate=lr,
                            max_depth=max_depth,
                            reg_alpha = reg_alpha,
                            reg_lambda = reg_lambda,
                            )        


        try:
            with mlflow.start_run(nested=True):
                
                if model == 'xgboost': 
                    selected_model = xb
                elif model == 'lightgbm':
                    selected_model = lb

                multi_output_gb = MultiOutputRegressor(selected_model)

                tscv = TimeSeriesSplit(n_splits=n_splits)
                scores = cross_val_score(xb, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
                scores = -scores
                mean_scores = scores.mean()
                rmse_val = np.sqrt(mean_scores)
                score_std = scores.std()

                multi_output_gb.fit(X_train, y_train)
                pred_test = multi_output_gb.predict(X_test)
                rmse_test = np.sqrt(mse(y_test, pred_test))

                mlflow.log_atr
                mlflow.log_param('split_type', split_type)
                mlflow.log_param('model', model)
                mlflow.log_param('n_estimators', n_estimators)
                mlflow.log_param('learning_rate', lr)
                mlflow.log_param('steps', steps)
                mlflow.log_param('max_depth', max_depth)
                mlflow.log_param('reg_alpha', reg_alpha)
                mlflow.log_param('reg_lambda', reg_lambda)

                mlflow.set_tag("Training Info", "GBM model for time-series")
                mlflow.log_metric('RMSE_val_set', rmse_val)
                mlflow.log_metric('std_dev', score_std)
                mlflow.log_metric('RMSE_test_set', rmse_test)

                signature = mlflow.models.infer_signature(X_train, 
                                                          multi_output_gb.predict(X_train),
                                                          params={'lr': lr,
                                                                  'n_estimators':n_estimators}
                                                          )

                model_info = mlflow.sklearn.log_model(
                    sk_model=multi_output_gb,
                    artifact_path="gb_model",
                    signature=signature,
                    input_example=X_train.iloc[:2],
                    registered_model_name="gb_model_registered"
                )
                
        except Exception as e:
            print(e)
        finally:
            mlflow.end_run()