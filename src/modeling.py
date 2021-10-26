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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: '%.2f' % x)


    

def run_model_cv(model_name, model, df, n_splits, n_repeats):
    
    np.random.seed(1991237)
    
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["ICU"]
    X = df.drop(["ICU",], axis=1)

    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats)

    result = cross_validate(model, X, y, cv=cv, scoring='roc_auc')

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

def roc_auc_rank(df: pd.DataFrame, param:dict, random_state: int = 0):
    """
    descr.

    Returns
    -------
    None.

    """
    y = df["ICU"]
    X = df.drop(["ICU"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        shuffle=True,
                                                        test_size=0.20,
                                                        random_state=random_state)

    roc_auc = []
    for feature in X_train.columns:

        # RandonForest_CLassiFier
        rf_clf = RandomForestClassifier(oob_score=param['oob_score'],
                                        n_estimators=param['n_estimators'],
                                        min_samples_split=param['min_samples_split'],
                                        min_samples_leaf=param['min_samples_leaf'],
                                        max_features=param['max_features'],
                                        max_depth=param['max_depth'],
                                        criterion=param['criterion'],
                                        bootstrap=param['bootstrap'])

        rf_clf.fit(X_train[feature].to_frame(), y_train)

        y_pred = rf_clf.predict(X_test[feature].to_frame())

        score = roc_auc_score(y_test, y_pred)

        roc_auc.append([feature, score])

    return pd.DataFrame(roc_auc, columns=['feature', 'roc_auc']).sort_values(by='roc_auc', ascending=False)



if __name__ == '__main__':
    
    df = pd.read_csv('../data/processed/df_featured.csv', index_col='Unnamed: 0')
    
    param = {'oob_score': False,
             'n_estimators': 2000,
             'min_samples_split': 10,
             'min_samples_leaf': 1,
             'max_features': 'auto',
             'max_depth': 30,
             'criterion': 'entropy',
             'bootstrap': False}
    f = roc_auc_rank(df, param)