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
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
    

def run_model_cv(model, df, n_splits, n_repeats):
    
    np.random.seed(1991237)
    
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["ICU"]
    X = df.drop(["ICU",], axis=1)

    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats)
    result = cross_validate(model, X, y, cv=cv, scoring='roc_auc')

    auc_mean = np.mean(result['test_score'])
    auc_std = np.std(result['test_score'])
    
    print(f'AUC Mean: {auc_mean.round(2)}\nAUC Std: {auc_std.round(3)}\nAUC CI: {(auc_mean - (2*auc_std)).round(2)} - {(auc_mean + (2*auc_std)).round(2)}')
    print('--------------------')
    
    return result['test_score']


if __name__ == '__main__':
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import cross_validate
    
    
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
    
    df = pd.read_csv('../data/interim/df_featured.csv', index_col='PATIENT_VISIT_IDENTIFIER')
    
    run_model_cv(LogisticRegression(max_iter=1000), df, 30, 15)
    
    run_model_cv(RandomForestClassifier( n_estimators=10), df, 30, 15)
    
    run_model_cv(HistGradientBoostingClassifier(), df, 30, 15)
    