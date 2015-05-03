Data

The community crime dataset has 1994 observations and 122 predictors. The goal is to predict per-capita crime rates in US communities using these predictors. Even though these predictors are normalized into the decimal range 0.00-1.00, they still retain their distributions and skewness. The entire dataset is randomly split into a training and test set. The training set consists of approximately 60% of the original data while the test set consists of approximately 40% of the original data. 


Script

The script learns 4 different models relying on the scikit-learn API written in Python. These models are LARS-based LASSO, linear SVM, random forest, and gradient boosting (gbm) regressors.  I used LASSO to reduce the number of predictors because I believe that many of them are correlated with one and another. To learn a linear SVM model, I only used the predictors previously selected by LASSO regression. Before learning regressors with LASSO and SVM, I standardized all the predictors. For random forest and gbm, I did not rely upon any feature selection or standardization. To learn all 4 models, I used 10 fold cross validation to optimize hyper-parameters. The score I used to optimize my hyper-parameters is mean squared error (MSE). MSE is also the metric I am using to evaluate my models. After each model is learned and the predictions for the test set are made by the corresponding models, the predictions are outputted into .csv files named "lasso_crime_predictions.csv", "linear_svm_crime_predictions.csv", "rf_crime_predictions.csv", and "gbm_crime_predictions.csv". 
