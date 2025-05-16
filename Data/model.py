'''Helper functions for the models'''
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold

# Functions -- Random Forest Classifier
def platform_preprocess(X_train, X_test):
    # preprocess data
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test

def platform_train_process(X_train, y_train):
    # model selection and training
    parameters_for_testing = {
    "n_estimators"    : [100,150,200] ,
     "max_features"        : [3,4],
    }
    model = RandomForestClassifier()
    kfold = KFold(n_splits=10, random_state=None)
    grid_cv = GridSearchCV(estimator=model, param_grid=parameters_for_testing, scoring='accuracy', cv=kfold)
    result = grid_cv.fit(X_train, y_train)
    print("Best: {} using {}".format(result.best_score_, result.best_params_))

    # model training
    tuned_model = RandomForestClassifier(n_estimators=result.best_params_['n_estimators'],
                                         max_features=result.best_params_['max_features'])
    tuned_model.fit(X_train, y_train)

    return tuned_model

def platform_test_model(model, X_test, y_test):
    # prediction on test data (benchmark)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# pp_X_train, pp_X_test = platform_preprocess(X_train, X_test)
# train
# model = platform_train_process(pp_X_train, y_train)
# print(str(model))

# # test
# accuracy = platform_test_model(model, pp_X_test, y_test)
# print("Acc: " + str(accuracy))

# Functions -- Logistic Regression
def platform_train_logreg(X_train1, y_train1):
    # hyperparameter grid
    param_grid = {
        "C":       [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        # use liblinear so both l1 & l2 are supported
        "solver":  ["liblinear"]
    }
    model = LogisticRegression(max_iter=1000)
    cv = KFold(n_splits=10, shuffle=True, random_state=101)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv
    )
    grid.fit(X_train1, y_train1)
    print(f"Best CV accuracy: {grid.best_score_:.4f} using {grid.best_params_}")

    # best_estimator_ already refits on full X_train
    return grid.best_estimator_

