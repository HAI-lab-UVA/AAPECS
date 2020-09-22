import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def impute_0(df, call_len, call_col, i):
    for j in range(call_len):
        df[call_col[j]].iloc[i] = 0

    return df


def impute_call(df):
    call_col1 = []
    call_col2 = []
    call_col3 = []
    col_names = df.columns
    for col_name in col_names:
        if 'callIncoming' in col_name:
            call_col1.append(col_name)
        if 'callOutgoing' in col_name:
            call_col2.append(col_name)
        if 'callMissed' in col_name:
            call_col3.append(col_name)

    call_len1 = len(call_col1)
    call_len2 = len(call_col2)
    call_len3 = len(call_col3)

    for i in range(len(df)):
        call_row1 = df[call_col1].iloc[i]
        if call_row1.isna().sum()/call_len1 >= 1:
            df = impute_0(df, call_len1, call_col1, i)
        call_row2 = df[call_col2].iloc[i]
        if call_row2.isna().sum()/call_len2 >= 1:
            df = impute_0(df, call_len2, call_col2, i)
        call_row3 = df[call_col3].iloc[i]
        if call_row3.isna().sum()/call_len3 >= 1:
            df = impute_0(df, call_len3, call_col3, i)

    return df


def impute_iter(df, i):
    imp = IterativeImputer(max_iter = 10, random_state = i * 60, sample_posterior = True)
    imp.fit(df)
    data_impute = imp.transform(df)
    df_impute = pd.DataFrame(data_impute, columns = df.columns)

    return df_impute
