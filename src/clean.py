# -*- coding: utf-8 -*-

"""
Created on Wed Sep 29 22:51:09 2021

@author: Filipi Rigui
"""

import pandas as pd
import numpy as np

def print_nan_count_by_feature(df: pd.DataFrame):
    '''
    Imprime a contagem de observações com o valor NaN sumarizado por feature

    Parameters
    ----------
    df : pd.DataFrame
        O DataFrame que se deseja sumarizar.

    Returns
    -------
    None.
    
    Examples
    --------
    >>> import sys
    >>> import pandas as pd
    >>> sys.path.insert(1, "../src")
    >>> from clean import print_nan_count_by_feature
    >>> df_path = 'https://github.com/fdrigui/covid19_icu_admission_prediction/raw/main/data/raw/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx'
    >>> df = pd.read_excel(df_path)
    >>> print_nan_count_by_feature(df)
    
    Count - Feature Name
    0000  - PATIENT_VISIT_IDENTIFIER
    0000  - AGE_ABOVE65
    0000  - AGE_PERCENTIL
    0000  - GENDER
    0005  - DISEASE GROUPING 1
    0005  - DISEASE GROUPING 2
    0005  - DISEASE GROUPING 3
    0005  - DISEASE GROUPING 4
    0005  - DISEASE GROUPING 5
    0005  - DISEASE GROUPING 6
    0005  - HTN
    ...
    '''
    
    # Imprime o cabeçalho
    print('Count - Feature Name\n--------------------')
    
    # Conta a quantidaded de NaN por feature
    nan_by_col = df.isna().sum()
    
    # Itera para cada feature, imprimindo a quantidade e o nome de cada uma
    for i in range(nan_by_col.size):
        label = nan_by_col.index[i]
        nan_count = nan_by_col[i]
        print(f'{str(nan_count).zfill(4)}  - {label}')


def fill_missing_data(df:pd.DataFrame, grouby:[str], method: str = None)->pd.DataFrame:
    '''
    Agrupa o Dataframe de entrada pela feature

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que se deseja aplicar o método de preenchimento
    grouby : [str]
        Usado para descrever por qual feature(s) o DataFrame vai ser agrupado
    method : {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}
        Método de preenchimento das colunas (Features) do DataFrame

    Returns
    -------
    filled_concatenated_dataset : TYPE
        Retorna dataframe preenchido conforme método de Filtro e agrupamento de entrada
    
    Examples
    --------
    >>> import sys
    >>> import pandas as pd
    >>> sys.path.insert(1, "../src")
    >>> from clean import print_nan_count_by_feature
    >>> df_path = 'https://github.com/fdrigui/covid19_icu_admission_prediction/raw/main/data/raw/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx'
    >>> df = pd.read_excel(df_path)
    >>> df_filled = fill_missing_data(df, 'PATIENT_VISIT_IDENTIFIER', 'bfill')
    >>> df_filled.head()

    '''
    continuos_feature_columns = df.iloc[:, 13:-2].columns
    filled_coltinuous_features = df.groupby(grouby, as_index=False)[continuos_feature_columns].fillna(method=method)
    categorical_features = df.iloc[:, :13]
    output = df.iloc[:, -2:]
    
    filled_concatenated_dataset = pd.concat([categorical_features, filled_coltinuous_features, output], ignore_index=True,axis=1)
    filled_concatenated_dataset.columns = df.columns
    
    return filled_concatenated_dataset


def neighborhood_missing_data(df:pd.DataFrame, grouby:[str])->pd.DataFrame:
    '''
    Preenche os valores NaN agrupando por 'PATIENT_VISIT_IDENTIFIER'.
    Primeiro faz o 'Backward fill', preenchendo para as lacunas de NaN para trás.
    Depois faz o 'Foward fill', preenchendo as lacunas de NaN para frente.

    Parameters
    ----------
    df : pd.DataFrame
        O DataFrame que se deseja aplicar o método de preenchimento

    Returns
    -------
    filled : pd.DataFrame
        DataFrame com os dados preenchidos
    
    Examples
    --------
    >>> import sys
    >>> import pandas as pd
    >>> sys.path.insert(1, "../src")
    >>> from clean import print_nan_count_by_feature
    >>> df_path = 'https://github.com/fdrigui/covid19_icu_admission_prediction/raw/main/data/raw/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx'
    >>> df = pd.read_excel(df_path)
    >>> df_without_nan = neighborhood_missing_data(df)
    >>> df_without_nan.head()

    '''
    
    # Faz o preenchimento para trás
    bfill = fill_missing_data(df, grouby, 'bfill')
    
    # Faz o preenchimento para frente
    filled = fill_missing_data(bfill, grouby, 'ffill')
    return filled


def drop_features_with_same_value_for_all_observations(df:pd.DataFrame, print_dropped_columns:bool= False)-> pd.DataFrame:
    '''
    Remove do DataFrame Features que contém um único valor, repetido em todas as observações

    Parameters
    ----------
    df : pd.DataFrame
        O DataFrame que se deseja demover as features com mesmo valor
    print_dropped_columns : bool, optional
        Imprime as colunas que foram removidas. The default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame sem fratures que repetem os valores em todas as observações
        
    Examples
    --------
    >>>

    '''
    unique = df.nunique() == 1
    if print_dropped_columns:
        for column in df.columns[unique]:
            print(column)
            
    print(f'Total of dropped columns: {sum(unique)}')
    
    return df.loc[:,~unique]

def plot_features_with_same_value_for_all_observations(df:pd.DataFrame)-> pd.DataFrame:
    '''
    Impime o nome das features que tem um único valor repetido em todas as observações

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que se deseja avaliar

    Examples
    --------
    >>>
    '''
    unique = df.nunique() == 1
    
    print('Nome das Colunas:\n--------------------')
    for i in range(len(unique)):
        if unique[i]:
            print(f'{unique.index[i]}')
    print(f'--------------------\nTotal: {unique[unique == True].size}')


def drop_duplicated_features(df:pd.DataFrame, print_dropped_columns:bool= False)-> pd.DataFrame:
    '''
    Avalia se duas ou mais features são identicas ( que tem valores iguais para cada observação),
    e retorna uma única feature, removendo as demais que são duplicadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que se deseja avaliar
    print_dropped_columns : bool, optional
        Mostra quais colunas foram removidas no processo de comparação de features

    Returns
    -------
    pd.DataFrame
        Retorna o DataFrame limpo, ou seja, com as features identicas removidas.

    '''
    
    duplicated_columns = df.T.duplicated()
    if print_dropped_columns:
        for column in df.columns[duplicated_columns]:
            print(column)
    
    print(f'Total dropped columns: {sum(duplicated_columns)}')
    
    return df.loc[:,~duplicated_columns]


def plot_duplicated_features(df:pd.DataFrame)-> pd.DataFrame:
    '''
    Imprime o nome de features que são identicas ( que tem valores iguais para cada observação).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que se deseja avaliar

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    duplicated_df = df.T.duplicated()
    duplicated_columns = df.loc[:,duplicated_df].columns
    
    print('Nome das Colunas:\n--------------------')
    for col in duplicated_columns:
        print(f'{col}')
    print(f'--------------------\nTotal: {duplicated_columns.size}')
    

def rename_portion_of_columns(df:pd.DataFrame, start:int, end: int, to_replace: str, value: str) -> pd.DataFrame:
    '''
    

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    start : int
        DESCRIPTION.
    end : int
        DESCRIPTION.
    to_replace : str
        DESCRIPTION.
    value : str
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    
    columns = df.columns.to_series()
    columns[start:end] = columns[start:end].str.replace(to_replace, value)
    df.columns = columns
    
    return df
    


def prepare_window(rows):
    '''
    descr

    Parameters
    ----------
    rows : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if(np.any(rows["ICU"])):
        rows.loc[rows["WINDOW"]=="0-2", "ICU"] = 1
    return rows.loc[rows["WINDOW"] == "0-2"]


if __name__ == '__main__':
    
    import pandas as pd
    
    df = pd.read_excel('https://github.com/fdrigui/covid19_icu_admission_prediction/raw/main/data/raw/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
    
    df_1_without_nan = neighborhood_missing_data(df, 'PATIENT_VISIT_IDENTIFIER')
    
    # print_nan_count_by_feature(df_1_without_nan)
    
    df_2_without_nan = df_1_without_nan.drop(df_1_without_nan.query('UREA_MEDIAN.isnull()', engine='python').index)
    
    # plot_features_with_same_value_for_all_observations(df_2_without_nan)
    
    df_3_without_same_value_col = drop_features_with_same_value_for_all_observations(df_2_without_nan, False)
    
    # plot_features_with_same_value_for_all_observations(df_3_without_same_value_col)
    
    # plot_duplicated_features(df_3_without_same_value_col)  
    
    df_4_without_duplicated_features = drop_duplicated_features(df_3_without_same_value_col, False)
    
    # plot_duplicated_features(df_4_without_duplicated_features)
    
    df_5_renamed = rename_portion_of_columns(df_4_without_duplicated_features, 13, (13+36), '_MEDIAN', '')
    
    # df_5_renamed.columns[13:13+36]