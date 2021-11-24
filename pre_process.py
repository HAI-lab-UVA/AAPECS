import os
import pandas as pd
import copy
from param import pers_names, col_infor, thres_par, thres_row, thres_col
from imputation import impute_call


def list_files(path):
    os.chdir(path)
    ls_files = os.listdir(path)

    return ls_files


def make_dir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    return path


def remove_har_sms(df):
    col_del = list()
    col_names = df.columns
    for col_name in col_names:
        col_name_split = col_name.split('_')
        ## bluetooth & application
        if 'recognition' in col_name_split or 'messages' in col_name_split:
            col_del.append(col_name)

    df_del = df.drop(col_del, axis=1)
    df_del.reset_index(drop=True, inplace=True)

    return df


def remove_ground_truth(df, pers_names):
    for pers_name in pers_names:
        row_del = list()
        for i, pers_val in enumerate(df[pers_name].tolist()):
            if pd.isna(pers_val):
                row_del.append(i)

        df = df.drop(row_del)
        df.reset_index(drop=True, inplace=True)

    return df


def remove_missing(df=None, remove_truth = True, remove_android = True, proc_call = True, remove_col = True, remove_row = False, remove_par = False):
    # Remove participants with missing ground truth
    if remove_truth:
        df = remove_ground_truth(df, pers_names)

    # Remove har & sms features
    if remove_android:
        df = remove_har_sms(df)

    # Fill missing in call features
    if proc_call:
        df = impute_call(df)

    # Removing each row with too much missing
    if remove_row:
        df_del = copy.deepcopy(df)
        df_del = df_del.drop(col_infor, axis = 1)
        df_del = df_del.drop(pers_names, axis=1)
        index_del = list()
        missing_rates = df_del.isnull().sum(axis=1) / df_del.shape[1]
        for i in range(df_del.shape[0]):
            if missing_rates[i] >= thres_row:
                index_del.append(i)

        df = df.drop(index_del)
        df.reset_index(drop=True, inplace=True)

    # Removing participants with less than 3 days' data
    if remove_par:
        row_del = list()
        df_group = df.groupby('participantID')
        df_group_summary = df_group.count()
        for i in range(len(df_group_summary)):
            if df_group_summary['local_segment'].iloc[i] < thres_par:
                row_del.append(df_group_summary.index[i])

        # print(row_del)
        df = df[~df['participantID'].isin(row_del)]
        df.reset_index(drop=True, inplace=True)

    # Removing features with too much missing
    if remove_col:
        col_del = list()
        col_names = df.columns.tolist()
        for i in pers_names:
            col_names.remove(i)
        #
        # for i in col_infor:
        #     col_names.remove(i)

        missing_rates = df[col_names].isnull().sum() / len(df)
        for i in range(len(missing_rates)):
            if missing_rates[i] >= thres_col:
                col_del.append(col_names[i])

        df = df.drop(col_del, axis=1)

        return df

    # df.reset_index(drop=True, inplace=True)

    # ls_ids = df['participantID'].tolist()
    # df.index = ls_ids
    col_drop = copy.deepcopy(col_infor)
    col_drop.remove('participantID')
    df = df.drop(col_drop, axis=1)
    # df.index = ls_ids
    # print(col_del)

    return df


# def pre_proc(path, file_name, row_thres):
#     os.chdir(path)
#     df_aware = pd.read_csv(file_name, sep = ',')
#
#     # Delete app data
#     col_app = []
#     col_names = df_aware.columns
#     for col_name in col_names:
#         if col_name[0:3] == 'sms' or col_name[0:3] == 'app':
#             col_app.append(col_name)
#
#     df_aware = df_aware.drop(col_app, axis=1)
#     df_aware.reset_index(drop=True, inplace=True)
#
#     # Delete partipants with little data
#     row_del = []
#     df_aware_group = df_aware.groupby('participantID')
#     df_aware_group_summary = df_aware_group.count()
#     for i in range(len(df_aware_group_summary)):
#         if df_aware_group_summary['localDate'].iloc[i] < row_thres:
#             row_del.append(df_aware_group_summary.index[i])
#
#     df_aware = df_aware[~df_aware['participantID'].isin(row_del)]
#     # df_aware = df_aware.drop(col_del, axis=1)
#     df_aware = df_aware.dropna(axis=0, how='all')
#     df_aware.reset_index(drop=True, inplace=True)
#     # df_aware_daily = filter_daily(df_aware)
#
#     return df_aware


def filter_daily(df):
    col_names = df.columns
    left_col_names = []
    # left_col_names.append('participantID')
    epoch = ['morning', 'afternoon', 'evening', 'night']
    for col_name in col_names:
        col_name_split = col_name.split('.')
        if col_name_split[-1] not in epoch:
            left_col_names.append(col_name)

    # left_col_names.extend(pers_names)
    df = df[left_col_names]

    return df