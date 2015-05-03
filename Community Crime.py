import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# output prediction files
def ouput_file(predictions, name):
    formatted_predictions=np.asarray(zip(range(1, predictions.size+1), predictions))
    file_name = name + '_predictions.csv'
    with open(file_name, 'w') as writer:
        writer.write('ID,Target\n')
        np.savetxt(writer, formatted_predictions, fmt='%.1f', delimiter=",")
        
np.random.seed(1)

# Community Crime
loc="/Users/Dan/OneDrive/Documents/Machine Learning/Virtual Machine/cs589vm/shared_files/data/HW2DataDistribute/CommunityCrime/"

training_set=loc + "train.npy"
testing_set=loc + "test_distribute.npy"

training=np.load(training_set)
testing=np.load(testing_set)[:,1:]

np.set_printoptions(suppress=True)

# Deleteing non-predictive features
training = np.delete(training, range(1, 5), 1)
testing = np.delete(testing, range(4), 1)

train_x=training[:,1:]
train_y=training[:,0]



## standardize dataset
scaler = preprocessing.StandardScaler().fit(train_x)
standardized_train_x = scaler.transform(train_x)
standardized_test_x = scaler.transform(testing)



# lars-based lasso with 10 fold cross validation
model = linear_model.LassoLarsCV(cv=10, normalize=False, n_jobs=-1).fit(standardized_train_x, train_y)
mses=model.cv_mse_path_.mean(axis=-1)
lasso_coef=model.coef_
predictions=np.around(model.predict(standardized_test_x), decimals=1)
ouput_file(predictions, "lasso_crime")


# use features selected by LASSO
nonzero_index=np.flatnonzero(lasso_coef)
lasso_train_x=standardized_train_x[:,nonzero_index]
lasso_test_x=standardized_test_x[:,nonzero_index]



# linear svm after LASSO
tuned_parameters = [{'C':[pow(10,i) for i in range(-2, 1)],
                    'epsilon': [0.1, 1]
                    }]

clf = GridSearchCV(SVR(kernel='linear'), tuned_parameters, cv=10, n_jobs=-1, scoring="mean_squared_error")
clf.fit(lasso_train_x, train_y)

predictions=clf.best_estimator_.predict(lasso_test_x)
predictions=np.around(predictions, decimals=1)
ouput_file(predictions, "linear_svm_crime")



# reload dataset so we have unpreprocessed data for random forest and gbm

training=np.load(training_set)
testing=np.load(testing_set)[:,1:]

train_x=training[:,1:]
train_y=training[:,0]


# random forest without any feature selection

tuned_parameters = [{'n_estimators':[100, 200, 300]}]

clf = GridSearchCV(RandomForestRegressor(n_jobs=-1, random_state=1), tuned_parameters, cv=10, n_jobs=-1, scoring="mean_squared_error")
clf.fit(train_x, train_y)
predictions=clf.best_estimator_.predict(testing)
predictions=np.around(predictions, decimals=1)
ouput_file(predictions, "rf_crime")



# gbm without any feature selection

tuned_parameters = [{'n_estimators':[1000, 1250],
                    'learning_rate': [pow(10,i) for i in range(-3, 0)]
                    }]

clf = GridSearchCV(GradientBoostingRegressor(subsample=0.5, random_state=1), tuned_parameters, cv=10, n_jobs=-1, scoring="mean_squared_error")
clf.fit(train_x, train_y)
predictions=clf.best_estimator_.predict(testing)
predictions=np.around(predictions, decimals=1)
ouput_file(predictions, "gbm_crime")
