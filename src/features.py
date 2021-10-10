# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:09:13 2021

@author: filip
"""


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression


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


def high_correlation_feature(df: pd.DataFrame, threshold: float = 0.95):
    """
    descr.

    Returns
    -------
    None.

    """
    # Compute the correlation matrix
    corr = df.iloc[:, 13: -2].corr().abs()
    masked_corr = corr.copy()
    masked_corr[corr < threshold] = 0
    masked_corr[corr >= threshold] = 1

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(masked_corr, mask=mask, cmap=cmap, center=0,
                square=True, vmin=0, vmax=1.0, cbar=False)

    upper_tri = masked_corr.where(np.triu(np.ones(masked_corr.shape), k=1)\
                                  .astype(bool))

    dunha = []
    for i in range(len(upper_tri)):
        for j in range(len(upper_tri)):
            if upper_tri.iloc[i, j] > threshold:
                dunha.append([upper_tri.columns[i], upper_tri.columns[j]])
                print(f'[{i}] {upper_tri.index[i]} x [{j}] {upper_tri.columns[j]}; corr: {corr.iloc[i, j].round(2)}')

    k = 0
    f, axes = plt.subplots((len(dunha)//3)+1, 3, figsize=(20, float(4*len(dunha)//3)+1))

    f.suptitle('Scatterplot Variáveis para ser excluidas devido alta correlação')
    for i, j in dunha:
        sns.scatterplot(data=df, x=i, y=j, hue='ICU', ax=axes[k//3, k%3])
        k+=1

    plt.show()

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

def residual_bloodpressure_sistolic_max_mean(df: pd.DataFrame) -> pd.DataFrame:
    x = df['BLOODPRESSURE_SISTOLIC_MEDIAN']
    x = x.to_numpy().reshape(-1, 1)
    y = df['BLOODPRESSURE_SISTOLIC_MAX']
    reg = LinearRegression()
    reg.fit(x,y)
    prediction = reg.predict(x)
    residual = (y - prediction)
    df_6_wt_bloodpressure_residual = df.copy()
    df_6_wt_bloodpressure_residual['BLOODPRESSURE_SISTOLIC_MAX'] = residual
    df_6_wt_bloodpressure_residual = df_6_wt_bloodpressure_residual.rename(columns={'BLOODPRESSURE_SISTOLIC_MAX':'BLOODPRESSURE_MAX_MEDIAN_RESIDUAL'})
    
    return df_6_wt_bloodpressure_residual

def feature(df: pd.DataFrame) -> pd.DataFrame:
    '''
    

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    df_5_wo_bloodpressure_special_cause : TYPE
        DESCRIPTION.

    '''
    df_1_agescaled = scale_age_percentil(df)
    df_2_wo_window = df_1_agescaled.drop('WINDOW', axis=1)
    df_3_wo_low_var_feat = drop_low_variance_feature(df_2_wo_window)
    df_4_wo_low_roc_auc = drop_univariate_roc_auc(df_3_wo_low_var_feat,
                                                  low_cutband=0.48,
                                                  high_cutband=0.52,
                                                  random_state=791723,
                                                  plot=False)
    
    df_5_wo_bloodpressure_special_cause = df_4_wo_low_roc_auc.drop(101)
    
    return df_5_wo_bloodpressure_special_cause


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

    # df_4_wo_low_roc_auc = drop_univariate_roc_auc(df_3_wo_low_var_feat,
    #                                               low_cutband=0.48,
    #                                               high_cutband=0.52,
    #                                               random_state=791723,
    #                                               plot=False)

    # plot_univariate_roc_auc(df_4_wo_low_roc_auc, low_cutband=0.48,
    #                         high_cutband=0.52, random_state=791723,
    #                         plot=True)
    
    high_correlation_feature(df_4_wo_low_roc_auc)
    
    df_5_wo_bloodpressure_special_cause = df_4_wo_low_roc_auc.drop(101)
