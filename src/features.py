# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:09:13 2021

@author: filip
"""


import pandas as pd
# import numpy as np


from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def normalization_zero_to_one(serie: pd.Series) -> pd.Series:
    """
    descr.

    Parameters
    ----------
    serie : pd.Series
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # return (serie - np.min(serie)) / np.ptp(serie)
    return ((serie - serie.min()) / (serie.max() - serie.min()))


def scale_age_percentil(df: pd.DataFrame) -> pd.DataFrame:
    """
    descr.

    Parameters
    ----------
    age_percentil : pd.Series
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    df['AGE_PERCENTIL'] = normalization_zero_to_one(age_percentil_float)

    df.rename({'AGE_PER
    """
    age_percentil_category = df['AGE_PERCENTIL']

    age_percentil_float = age_percentil_category.str.replace('th', '')\
                                                .replace('Above 90', '100')\
                                                .astype('float')

    df['AGE_PERCENTIL'] = normalization_zero_to_one(age_percentil_float)

    df = df.rename({'AGE_PERCENTIL': 'AGE_SCALED'}, axis=1)

    return df


def plot_low_variance_feature(df: pd.DataFrame, threshold: float = 0.01,
                              random_state: int = 0):
    """
    desc.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    y = df["ICU"]
    X = df.drop(["ICU"], axis=1)

    vt = VarianceThreshold(threshold)
    _ = vt.fit(X, y)

    mask = vt.get_support()

    print(X.loc[:, ~mask].columns)


def drop_low_variance_feature(df: pd.DataFrame, threshold: float = 0.01,
                              random_state: int = 0) -> pd.DataFrame:
    """
    desc.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    y = df["ICU"]
    X = df.drop(["ICU"], axis=1)

    vt = VarianceThreshold(threshold)
    _ = vt.fit(X, y)

    mask = vt.get_support()

    df_wto_low_var_feat = X.loc[:, mask]

    df_wo_low_var_feat_wt_y = pd.concat([df_wto_low_var_feat, y], axis=1)

    return df_wo_low_var_feat_wt_y


def plot_univariate_roc_auc(df: pd.DataFrame, low_cutband: float,
                            high_cutband: float, random_state: int = 0,
                            plot: bool = True):
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
        rf_clf = RandomForestClassifier(n_estimators=500,
                                        random_state=random_state)

        rf_clf.fit(X_train[feature].to_frame(), y_train)

        y_pred = rf_clf.predict(X_test[feature].to_frame())

        score = roc_auc_score(y_test, y_pred)

        if (score >= low_cutband) & (score <= high_cutband):
            roc_auc.append(feature)

            if plot:
                print('Feature:', feature, '; ROC:', score.round(3))

    print('--------------------')
    print(f'Foram encontradas {len(roc_auc)} features com ROC AOC entre {low_cutband} e {high_cutband}')

    return roc_auc


def drop_univariate_roc_auc(df: pd.DataFrame, low_cutband: float,
                            high_cutband: float, random_state: int = 0,
                            plot: bool = False) -> pd.DataFrame:
    """
    descr.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    low_cutband : float
        DESCRIPTION.
    high_cutband : float
        DESCRIPTION.
    random_state : int, optional
        DESCRIPTION. The default is 0.
    plot : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    roc_auc = plot_univariate_roc_auc(df=df, low_cutband=low_cutband,
                                      high_cutband=high_cutband,
                                      random_state=random_state, plot=plot)

    df_wt_low_roc_auc = df.drop(roc_auc, axis=1)

    return df_wt_low_roc_auc


if __name__ == '__main__':

    import pandas as pd

    df = pd.read_csv('../data/interim/df_7_cleaned.csv',
                     index_col='PATIENT_VISIT_IDENTIFIER')

    df_1_agescaled = scale_age_percentil(df)

    df_2_wo_window = df_1_agescaled.drop('WINDOW', axis=1)

    # plot_low_variance_feature(df_2_wo_window)

    df_3_wo_low_var_feat = drop_low_variance_feature(df_2_wo_window)

    # plot_low_variance_feature(df_3_wo_low_var_feat)

    # plot_univariate_roc_auc(df_3_wo_low_var_feat, low_cutband=0.48,
    #                        high_cutband=0.52, random_state=791723,
    #                        plot=True)

    df_4_wo_low_roc_auc = drop_univariate_roc_auc(df_3_wo_low_var_feat,
                                                  low_cutband=0.48,
                                                  high_cutband=0.52,
                                                  random_state=791723,
                                                  plot=False)

    # plot_univariate_roc_auc(df_4_wo_low_roc_auc, low_cutband=0.48,
    #                         high_cutband=0.52, random_state=791723,
    #                         plot=True)
