# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 21:57:23 2021

@author: filipi
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestCentroid
from supervised import LazyClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: '%.2f' % x)


    

def run_model_cv(model_name, model, df, n_splits, n_repeats):
    
    np.random.seed(1991237)
    
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["ICU"]
    X = df.drop(["ICU",], axis=1)

    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats)
    
    if model_name.lstrip(' ') == 'RandomForestClassifier':

        clf = model(max_depth=10, random_state=0)
        result = cross_validate(clf, X, y, cv=cv, scoring='roc_auc')
        
    elif model_name.lstrip(' ') == 'xgb':
        
        clf = xgb.XGBRegressor(n_estimators =10, objective='binary:logistic',eval_metric='logloss')
        result = cross_validate(clf, X, y, cv=cv, scoring='roc_auc')
        
    elif model_name.lstrip(' ') == 'DummyClassifier':
        clf = DummyClassifier(strategy="prior")
        result = cross_validate(clf, X, y, cv=cv, scoring='roc_auc')
    
    else:      
        clf = model()
        result = cross_validate(clf, X, y, cv=cv, scoring='roc_auc')

    auc_mean = np.mean(result['test_score']).round(2)
    auc_std = np.std(result['test_score']).round(3)
    
    print(f'{model_name}: AUC Mean: {auc_mean}, AUC Std: {auc_std.round(3)}, AUC CI: {(auc_mean - (2*auc_std)).round(2)} - {(auc_mean + (2*auc_std)).round(2)}')


def many_Lazy_Classifiers(df: pd.DataFrame, n:int):
    
    np.random.seed(1991237)
    
    y = df["ICU"]
    X = df.drop(["ICU"], axis=1)
    
    model_list = []
    for _ in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True)
        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
        models,predictions = clf.fit(X_train, X_test, y_train, y_test)
        model_list.append(models['ROC AUC'])
        
    return pd.DataFrame([pd.DataFrame(model_list).mean(axis=0),
                         pd.DataFrame(model_list).std(axis=0)],
                        index=['mean','std']).T.sort_values(by='mean', ascending=False)



if __name__ == '__main__':
    
    df = pd.read_csv('../data/processed/df_featured.csv', index_col='Unnamed: 0')
    classifier_rank = many_Lazy_Classifiers(df, 30)
