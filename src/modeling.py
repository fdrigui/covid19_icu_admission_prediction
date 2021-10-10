# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 21:57:23 2021

@author: filip
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: '%.2f' % x)


    

def run_model_cv(model_name, model, df, n_splits, n_repeats):
    
    np.random.seed(1991237)
    
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["ICU"]
    X = df.drop(["ICU",], axis=1)

    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats)
    
    if model_name == '         RandomForestClassifier':

        clf = model(max_depth=10, random_state=0)
        result = cross_validate(clf, X, y, cv=cv, scoring='roc_auc')
        
    elif model_name == '                            xgb':
        
        clf = xgb.XGBRegressor(n_estimators =10, objective='binary:logistic',eval_metric='logloss')
        result = cross_validate(clf, X, y, cv=cv, scoring='roc_auc')
    
    else:      
        clf = model()
        result = cross_validate(clf, X, y, cv=cv, scoring='roc_auc')

    auc_mean = np.mean(result['test_score'])
    auc_std = np.std(result['test_score'])
    
    print(f'{model_name}: AUC Mean: {auc_mean}, AUC Std: {auc_std.round(3)}, AUC CI: {(auc_mean - (2*auc_std))} - {(auc_mean + (2*auc_std))}')



def run__classifier_list(df: pd.DataFrame):
    
    removed_classifiers = [
    ('         RandomForestClassifier', RandomForestClassifier),
    ('     GradientBoostingClassifier', GradientBoostingClassifier),
    ('      GaussianProcessClassifier', GaussianProcessClassifier),
    (' HistGradientBoostingClassifier', HistGradientBoostingClassifier),
    ('                  MLPClassifier', MLPClassifier),
    ('           LogisticRegressionCV', LogisticRegressionCV),
    ('                            xgb', xgb)]
    
    for model_name, model in removed_classifiers:
        run_model_cv(model_name, model, df, 5, 5)


if __name__ == '__main__':
    
    df = pd.read_csv('../data/processed/df_featured.csv', index_col='Unnamed: 0')
    
    clf = KNeighborsClassifier
    run_model_cv('KNeighborsClassifier', clf, df, 10, 10)
    
    #run__classifier_list(df)

