# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:09:13 2021

@author: filip
"""
import pandas as pd
#import numpy as np


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

if __name__ == '__main__':
    
    import pandas as pd
    
    df = pd.read_csv('../data/interim/df_7_cleaned.csv')
    
    df_1_agescaled = scale_age_percentil(df)