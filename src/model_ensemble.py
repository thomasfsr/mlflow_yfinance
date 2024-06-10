from sklearn.ensemble import GradientBoostingRegressor as gbm
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, KFold

class GbmModel:
    def __init__(self, df:pd.DataFrame = None,
                 n_X:str = 15, 
                 splits:str = 5, 
                 rs:int = 42, 
                 ts:str= .2):
        
        self.df = df
        self.splits = splits
        self.n_X = n_X
        self.rs = rs
        self.ts = ts

    def splitting(self):
        df = self.df.copy() 
        n_X = self.n_X
        ts = self.ts
        rs = self.rs

        vol = df['vol'].copy()
        df.drop('vol', axis=1,inplace=True)
        X = df.iloc[:,:n_X].copy()
        X['vol'] = vol
        y = df.iloc[:, n_X:].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y,  
                                                            test_size=ts, 
                                                            shuffle=True, 
                                                            random_state=rs)
        return X_train, X_test, y_train, y_test

    def model(self, X_train, y_train, lr: float, n_estimators: int):
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

                mlflow.log_param('n_estimators', n_estimators)
                mlflow.log_param('learning_rate', lr)
                mlflow.set_tag("Training Info", "GBM model for time-series")
                mlflow.log_metric('mean_squared_error', mean_scores)
                mlflow.log_metric('std_dev', score_std)
                
                multi_output_gb.fit(X_train, y_train)
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

        except Exception as e:
            print(e)
        finally:
            mlflow.end_run()

