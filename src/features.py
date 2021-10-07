# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:09:13 2021

@author: filip
"""


import pandas as pd
#import numpy as np


from sklearn.feature_selection import VarianceThreshold



def normalization_zero_to_one(serie: pd.Series) -> pd.Series:
    '''
    descr

    Parameters
    ----------
    serie : pd.Series
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # return (serie - np.min(serie)) / np.ptp(serie)
    return ((serie - serie.min()) / (serie.max() - serie.min()))

def scale_age_percentil(df:pd.DataFrame)-> pd.DataFrame:
    '''
    descr

    Parameters
    ----------
    age_percentil : pd.Series
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
                          .replace('Above 90', '100')\
                                                 .astype('float64')
    
    df['AGE_PERCENTIL'] = normalization_zero_to_one(age_percentil_float)
    
    df.rename({'AGE_PER
    '''
    age_percentil_category = df['AGE_PERCENTIL']
    
    age_percentil_float = age_percentil_category.str.replace('th', '')\
                                                .replace('Above 90', '100')\
                                                .astype('float')
                                                
    df['AGE_PERCENTIL'] = normalization_zero_to_one(age_percentil_float)
    
    df = df.rename({'AGE_PERCENTIL':'AGE_SCALED'}, axis=1)
    
    return df


def plot_low_variance_feature(df: pd.DataFrame, threshold: float = 0.01,
                              random_state: int = 0)-> pd.DataFrame:
    '''
    desc

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    df = df.sample(frac=1,random_state=random_state).reset_index(drop=True)
    y = df["ICU"]
    X = df.drop(["ICU",], axis=1)
    
    vt = VarianceThreshold(threshold)
    _ = vt.fit(X, y)

    mask = vt.get_support()

    print(X.loc[:, ~mask].columns)

    
def drop_low_variance_feature(df: pd.DataFrame, threshold: float = 0.01,
                              random_state: int = 0)-> pd.DataFrame:
    '''
    desc

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    df = df.sample(frac=1,random_state=random_state).reset_index(drop=True)
    y = df["ICU"]
    X = df.drop(["ICU",], axis=1)
    
    vt = VarianceThreshold(threshold)
    _ = vt.fit(X, y)

    mask = vt.get_support()
    
    df_wto_low_var_feat = X.loc[:, mask]
    
    df_wo_low_var_feat_wt_y = pd.concat([df_wto_low_var_feat, y], axis=1)

    return df_wo_low_var_feat_wt_y



if __name__ == '__main__':
    
    import pandas as pd
    
    df = pd.read_csv('../data/interim/df_7_cleaned.csv', index_col='PATIENT_VISIT_IDENTIFIER')
    
    df_1_agescaled = scale_age_percentil(df)
    
    df_2_wo_window = df_1_agescaled.drop('WINDOW', axis=1)
    
    #plot_low_variance_feature(df_2_wo_window)
    
    df_3_wo_low_var_feat = drop_low_variance_feature(df_2_wo_window)
    
    plot_low_variance_feature(df_3_wo_low_var_feat)