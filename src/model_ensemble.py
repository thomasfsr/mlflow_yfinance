from sklearn.ensemble import GradientBoostingRegressor as gbm
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error as mse

class GbmModel:
    def __init__(self, df:pd.DataFrame = None,
                 n_X:str = 15, 
                 splits:str = 5, 
                 rs:int = 42, 
                 ts:str= .2,
                 val:pd.DataFrame = None,
                 vol:pd.DataFrame = None
                 ):
        
        self.df = df
        self.splits = splits
        self.n_X = n_X
        self.rs = rs
        self.ts = ts
        self.val = val
        self.vol = vol

    def splitting(self):
        if self.df is not None:        
            df = self.df.copy()
            n_X = self.n_X
            ts = self.ts
            rs = self.rs

            vol = df['vol'].copy()
            df.drop('vol', axis=1, inplace=True)
            X = df.iloc[:, :n_X].copy()
            X['vol'] = vol
            y = df.iloc[:, n_X:].copy()
            #print(X.iloc[:2])
            #print(y.iloc[:2])

        elif self.val is not None and self.vol is not None:
            n_X = self.n_X
            ts = self.ts
            rs = self.rs
            val = self.val.copy()
            vol = self.vol.copy()

            X_val = val.iloc[:, :n_X].copy()
            X_vol = vol.iloc[:, :n_X].copy()
            X = pd.concat([X_val, X_vol], axis=1)

            y = val.iloc[:, n_X:].copy()
            
            #print(X.iloc[:2])
            #print(y.iloc[:2])
        else:
            ValueError()
        X_train, X_test, y_train, y_test = train_test_split(X, y,  
                                                    test_size=ts, 
                                                    shuffle=True, 
                                                    random_state=rs)

        return X_train, X_test, y_train, y_test

    def model(self, lr: float, n_estimators: int):
        
        X_train, X_test, y_train, y_test = self.splitting()

        n_splits = self.splits
        rs = self.rs

        try:
            with mlflow.start_run():

                gbm_obj = gbm(n_estimators=n_estimators, learning_rate=lr, random_state=rs)
                multi_output_gb = MultiOutputRegressor(gbm_obj)

                kfold = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
                scores = cross_val_score(multi_output_gb, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

                scores = -scores
                mean_scores = scores.mean()
                score_std = scores.std()

                multi_output_gb.fit(X_train, y_train)
                pred_test = multi_output_gb.predict(X_test)
                rmse = np.sqrt(mse(y_test, pred_test))

                mlflow.log_param('n_estimators', n_estimators)
                mlflow.log_param('learning_rate', lr)
                mlflow.set_tag("Training Info", "GBM model for time-series")
                mlflow.log_metric('mean_squared_error', mean_scores)
                mlflow.log_metric('std_dev', score_std)
                mlflow.log_metric('RMSE_test_set', rmse)

                signature = mlflow.models.infer_signature(X_train, multi_output_gb.predict(X_train))

                model_info = mlflow.sklearn.log_model(
                    sk_model=multi_output_gb,
                    artifact_path="gbm_model",
                    signature=signature,
                    input_example=X_train.iloc[:2],
                    registered_model_name="gbm_model_registered"
                )
                print(f"Mean MSE for n_estimators={n_estimators} and learning_rate={lr}: {mean_scores}")
                print(f"Model Info: {model_info}")
                print(f"RMSE of test-set: {rmse}")

        except Exception as e:
            print(e)
        finally:
            mlflow.end_run()

