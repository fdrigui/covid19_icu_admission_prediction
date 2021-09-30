# -*- coding: utf-8 -*-

"""
Created on Wed Sep 29 22:51:09 2021

@author: Filipi Rigui
"""

import pandas as pd

def print_nan_count_by_feature(df: pd.DataFrame):
    '''
    descroption

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.
    
    Examples
    --------
    >>> from lazypredict.Supervised import LazyClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> X = data.data
    >>> y= data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    >>> clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    >>> models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    >>> models

    '''
    nan_by_col = df.isna().sum()
    for i in range(nan_by_col.size):
        label = nan_by_col.index[i]
        nan_count = nan_by_col[i]
        print(f'{str(nan_count).zfill(4)} - {label}')
        
        
def fill_missing_data(df:pd.DataFrame, method:str)->pd.DataFrame:
    '''
    description

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    method : str
        DESCRIPTION.

    Returns
    -------
    filled_concatenated_dataset : TYPE
        DESCRIPTION.

    '''
    continuos_feature_columns = df.iloc[:, 13:-2].columns
    filled_coltinuous_features = df.groupby("PATIENT_VISIT_IDENTIFIER", as_index=False)[continuos_feature_columns].fillna(method=method)
    categorical_features = df.iloc[:, :13]
    output = df.iloc[:, -2:]
    
    filled_concatenated_dataset = pd.concat([categorical_features, filled_coltinuous_features, output], ignore_index=True,axis=1)
    filled_concatenated_dataset.columns = df.columns
    
    return filled_concatenated_dataset

def neighborhood_missing_data(df:pd.DataFrame)->pd.DataFrame:
    '''
    descr.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    ffill : TYPE
        DESCRIPTION.

    '''
    bfill = fill_missing_data(df, 'bfill')
    ffill = fill_missing_data(bfill, 'ffill')
    return ffill
    
    