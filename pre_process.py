import os
import pandas as pd


def list_files(path):
    os.chdir(path)
    ls_files = os.listdir(path)

    return ls_files


def make_dir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    return path


def pre_proc(path, file_name, row_thres):
    os.chdir(path)
    df_aware = pd.read_csv(file_name, sep = ',')

    # Delete app data
    col_app = []
    col_names = df_aware.columns
    for col_name in col_names:
        if col_name[0:3] == 'sms' or col_name[0:3] == 'app':
            col_app.append(col_name)
        df_aware = df_aware.drop(col_app, axis = 1)

    # Delete partipants with little data
    row_del = []
    df_aware_group = df_aware.groupby('participantID')
    df_aware_group_summary = df_aware_group.count()
    for i in range(len(df_aware_group_summary)):
        if df_aware_group_summary['localDate'].iloc[i] < row_thres:
            row_del.append(df_aware_group_summary.index[i])

    df_aware = df_aware[~df_aware['participantID'].isin(row_del)]
    df_aware = df_aware.dropna(axis = 0, how = 'all')

    return df_aware