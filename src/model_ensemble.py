from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error as mse
    
class GbmModel:
    def __init__(self,
                 val:pd.DataFrame = None,
                 vol:pd.DataFrame = None,
                 n_X:str = 20, 
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
            #rs = self.rs
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
                                                    shuffle=False, 
                                                    #random_state=rs
                                                    )

        return X_train, X_test, y_train, y_test

    def splitting_sum_vol(self):
        if self.val is not None and self.vol is not None:
            n_X = self.n_X
            ts = self.ts
            #rs = self.rs
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
                                                    #shuffle=False, 
                                                    #random_state=rs
                                                    )

        return X_train, X_test, y_train, y_test

    def model(self, lr: float, n_estimators: int, split_type: str = 'all_vols'):

        assert split_type in ['all_vols', 'sum_vols'], "split_type must be 'all_vols' or 'sum_vols'"

        function_mapping = {
            'all_vols': self.splitting,
            'sum_vols': self.splitting_sum_vol
        }

        split_function = function_mapping.get(split_type)

        X_train, X_test, y_train, y_test = split_function()

        n_splits = self.splits
        rs = self.rs

        try:
            with mlflow.start_run():

                gbr_obj = gbr(n_estimators=n_estimators, 
                              random_state=rs, 
                              learning_rate=lr)

                multi_output_gb = MultiOutputRegressor(gbr_obj)

                #kfold = KFold(n_splits=n_splits, shuffle=False, 
                #              #random_state=rs
                #              )
                #scores = cross_val_score(multi_output_gb, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

                #scores = -scores
                #mean_scores = scores.mean()
                #score_std = scores.std()

                tscv = TimeSeriesSplit(n_splits=n_splits)
                scores = cross_val_score(multi_output_gb, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
                scores = -scores
                mean_scores = scores.mean()
                rmse_val = np.sqrt(mean_scores)
                score_std = scores.std()

                multi_output_gb.fit(X_train, y_train)
                pred_test = multi_output_gb.predict(X_test)
                rmse_test = np.sqrt(mse(y_test, pred_test))

                mlflow.log_param('split_type', split_type)
                mlflow.log_param('n_estimators', n_estimators)
                mlflow.log_param('learning_rate', lr)
                mlflow.set_tag("Training Info", "GBM model for time-series")
                mlflow.log_metric('RMSE_val_set', rmse_val)
                mlflow.log_metric('std_dev', score_std)
                mlflow.log_metric('RMSE_test_set', rmse_test)

                signature = mlflow.models.infer_signature(X_train, multi_output_gb.predict(X_train))

                model_info = mlflow.sklearn.log_model(
                    sk_model=multi_output_gb,
                    artifact_path="gbm_model",
                    signature=signature,
                    input_example=X_train.iloc[:2],
                    registered_model_name="gbm_model_registered"
                )
                print(f"Squared Root of Mean RMSE for n_estimators={n_estimators} and learning_rate={lr}: {mean_scores}")
                print(f"Model Info: {model_info}")
                print(f"RMSE of test-set: {rmse_test}")

        except Exception as e:
            print(e)
        finally:
            mlflow.end_run()

