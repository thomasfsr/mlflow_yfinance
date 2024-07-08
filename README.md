# MLflow to time series data  
  
## Overview  
MLflow is a well-established tool for monitoring ML models.  
This project aims to provide a hands-on application of this tool to monitor the pipeline of a model that will predict the price of any stock available in the YFinance library.  
In this case, I used the "close" price and volume of the asset for simplicity.   
  
## Transformation  
I created lag columns of the close price following the parameters of the class I created to prepare the dataset for use. The default number of lags was set to 30, resulting in the dataset being organized into sequences of 30 consecutive close prices and their respective volumes:  
  
First 5 lines of input data:  
| Date                |   Open |   High |    Low |   Close |   Adj Close |           Volume |
|:--------------------|-------:|-------:|-------:|--------:|------------:|-----------------:|
| 2000-08-23 00:00:00 |  31.95 |  32.8  |  31.95 |   32.05 |       32.05 |  79385           |
| 2000-08-24 00:00:00 |  31.9  |  32.24 |  31.4  |   31.63 |       31.63 |  72978           |
| 2000-08-25 00:00:00 |  31.7  |  32.1  |  31.32 |   32.05 |       32.05 |  44601           |
| 2000-08-28 00:00:00 |  32.04 |  32.92 |  31.86 |   32.87 |       32.87 |  46770           |
| 2000-08-29 00:00:00 |  32.82 |  33.03 |  32.56 |   32.72 |       32.72 |  49131           |
  
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
  
The number of days ahead to be predicted is also tunable and the default is 5 days out of those 30 lag days.  
For instance:  
|    |    t0 |    t1 |    t2 |    t3 |    t4 |   ... |   t22 |   t23 |   t24 |     sum_vol |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------------:|
|  0 | 32.05 | 31.63 | 32.05 | 32.87 | 32.72 | ...  | 31.57 | 31.5  | 31.5  | 1.93572e+06 |
|  1 | 33.4  | 33.1  | 33.38 | 33.8  | 34.95 | ...  | 32.15 | 32.07 | 31.5  | 1.94394e+06 |
|  2 | 35.33 | 33.7  | 35.1  | 34.2  | 33.8  | ...  | 31.88 | 33.15 | 33.15 | 1.98475e+06 |
|  3 | 34.1  | 35.85 | 36.88 | 36.5  | 37.5  | ...  | 33.1  | 32.95 | 33.55 | 1.84165e+06 |
|  4 | 33.95 | 32.65 | 31.57 | 31.5  | 31.5  | ...  | 33.73 | 33.37 | 32.96 | 1.72553e+06 |
  
##  Training  
The models choosen for this project are XGBoost and LightGBM because of the capacity if these models to capture complex patterns. It is achieved because GBM is a ensamble method, meaning that it is composed of multiple decision trees in series (different than random forest). Each model is trained to correct the erros of the previous decision tree.  
  
### Hyperparameters used:
- Number of Estimators:  
The number of trees (models) that will be built in the boosting process. Each tree is built sequentially, and the predictions from all trees are combined to make the final prediction.
  
- Learning Rate:  
Learning rate (also known as shrinkage) is a hyperparameter that controls the contribution of each tree (or weak learner) to the ensemble. It scales the impact of each tree's prediction on the final prediction.  
  
- Reg Alpha and Lambda:  
reg_alpha and reg_lambda are regularization parameters that control the complexity of the model to prevent overfitting.  
Specifically, reg_alpha (also known as L1 regularization) penalizes the absolute weights of features, encouraging sparsity in the model by pushing weights towards zero. This helps in feature selection by reducing less important features' weights effectively to zero. On the other hand, reg_lambda (L2 regularization) penalizes the squared weights of features, which discourages large weights and promotes smoother models by preventing individual feature weights from becoming too large. These regularization parameters are crucial in optimizing model performance by balancing between model complexity and generalization ability, thereby improving model robustness and preventing overfitting on training data. Adjusting these parameters allows fine-tuning of the model to achieve better performance on unseen data and improve overall predictive accuracy.  
  
### Resampling:  
To better assess the model's performance, I implemented time series cross-validation, which involves splitting the entire dataset into different folds. However, since it is s time series, a normal CV would create data leakage, so the solution was to implement TimeSeriesCV. Additionally, a portion of unseen data called Test set is held out until the end of the training process, enabling me to make predictions on the test set to evaluate the model's performance.  


### Evaluation:  
For CV, I chose 'neg_mean_squared_error' as the metric because it supports multi-output scenarios effectively.  
  
And to be tracked by MLflow I also used the average of the score and standard deviation.  
  
```python
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
```
  
In pipeline of training I also created 2 types of splitting:  
- All volume data: The columns of volume of the timestamp of training set was merged with the values to improve the prediction.  
- Sum of volumes: It was created one column with the sum of volumes of the training set.
Whether the sum of volumes would be sufficient to improve the model or not will be tested.  
  
### MLFlows logs:  
The parameters choosen to be track is:  
```python
mlflow.log_param('split_type', split_type)
mlflow.log_param('model', model)
mlflow.log_param('n_estimators', n_estimators)
mlflow.log_param('learning_rate', lr)
mlflow.log_param('steps', steps)
mlflow.log_param('max_depth', max_depth)
mlflow.log_param('reg_alpha', reg_alpha)
mlflow.log_param('reg_lambda', reg_lambda)
```
  
The config.yml file with the parameters:  
  
```yaml
parameters:
  learning_rate: [0.1, 0.5]
  n_estimators: [100]
  split_type: ['sum_vols']
  max_depth: [3, 10]
  steps: [10]
  reg_alpha: [0, .5]
  reg_lambda: [0, .5]
  model: ['lightgbm', 'xgboost']
```
  
## Training Script
  
There was created a grid of combinations of the hyperparameters during training:  
```python
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
mlflow.create_experiment("oil_pred")
mlflow.set_experiment("oil_pred")
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
```
  
## Results  
Now we can use mlflow ui to see the results:  
```bash
mlflow ui
```
Accessing http://127.0.0.1:5000/ we can see the experiments ran with mlflow.  
  
MLflow provide tools to create chart, specially to display the combination of the parameters related to a metric.  
![MLflow Chart](chart_mlflow.png)  

The best model can be seem in this line:  
![MLflow Chart Opt](opt_model.png)  
  
By this experiment, the best pair of parameters was:  
- Model: XGBoost  
- max_depth: 3  
- Learning Rate: 0.1  
- N_estimators: 100  
- Split type: Sum of Volumes  
- reg_alpha: 0.5  
- reg_lambda: 0  
  
Root Mean Squared Error for Test Set was 3.28.  
  
Morever, MLflow logs some artefacts of the experiment, such as portion of input data and the output, schema of the dataset:  
![artifacts](artifacts.png)  
  
It also provides a way to implement the model with spark:  
![Spark](spark.png)  
  
See also charts of some exemples of predictions made on test-set comparing to real data:  
![chart_pred](pred.png)  
  
## Conclusion:  
MLflow is a powerful tool for monitoring model performance, tuning hyperparameters, comparing models, and evaluating data drift that could degrade model performance, necessitating model re-training.  






