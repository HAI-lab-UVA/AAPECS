import pandas as pd
import os
import numpy as np
import pandas as pd


def row_diff(df):
    #validHours_value = df['validHours'].iloc[0]
    participantID_value = df['participantID'].iloc[0]
    ls_df = []
    col_names = df.columns
    for i in range(len(df) - 1):
        ls_df.append(list(df.iloc[i + 1] - df.iloc[i]))

    df_re = pd.DataFrame(ls_df, columns = col_names)

    #final_df['validHours'] = validHours_value
    df_re['participantID'] = participantID_value

    return df_re


def intra_feature(df):
    col_names = df.columns
    sel_col_names = []
    for col_name in col_names:
        sname = col_name.split('.')
        if sname[-1] == 'morning':
            sel_col_names.append([col_name, col_name[0:(len(col_name) - 8)], sname[-1]])
        if sname[-1] == 'afternoon':
            sel_col_names.append([col_name, col_name[0:(len(col_name) - 10)], sname[-1]])
        if sname[-1] == 'evening':
            sel_col_names.append([col_name, col_name[0:(len(col_name) - 8)], sname[-1]])
        if sname[-1] == 'night':
            sel_col_names.append([col_name, col_name[0:(len(col_name) - 6)], sname[-1]])

    df_sel_col_names = pd.DataFrame(sel_col_names, columns = ['col_names', 'feature', 'time'])

    return df_sel_col_names


def intra_feature_pairwise(df_sel_col_names, df):
    df_col_groups = df_sel_col_names.groupby('feature')
    df_col_group_names = df_col_groups.size().index
    for name in df_col_group_names:
        col_names_day = df_col_groups.get_group(name)
        col_names_day_time = list(col_names_day['time'])
        pos_morning = np.flatnonzero(col_names_day['time'] == 'morning')
        pos_afternoon = np.flatnonzero(col_names_day['time'] == 'afternoon')
        pos_evening = np.flatnonzero(col_names_day['time'] == 'evening')
        pos_night = np.flatnonzero(col_names_day['time'] == 'night')
        flag_morning = False
        flag_afternoon = False
        flag_evening = False
        flag_night = False
        if 'morning' in col_names_day_time and len(pos_morning) == 1:
            flag_morning = True
        if 'afternoon' in col_names_day_time and len(pos_afternoon) == 1:
            flag_afternoon = True
        if 'evening' in col_names_day_time and len(pos_evening) == 1:
            flag_evening = True
        if 'night' in col_names_day_time and len(pos_night) == 1:
            flag_night = True

        if (flag_morning and flag_afternoon):
            df[col_names_day['feature'].iloc[pos_morning[0]] + '.' + 'morning' + '-' + 'afternoon'] = df[col_names_day['col_names'].iloc[pos_morning[0]]] - df[col_names_day['col_names'].iloc[pos_afternoon[0]]]
        if (flag_morning and flag_evening):
            df[col_names_day['feature'].iloc[pos_morning[0]] + '.' + 'morning' + '-' + 'evening'] = df[col_names_day['col_names'].iloc[pos_morning[0]]] - df[col_names_day['col_names'].iloc[pos_evening[0]]]
        if (flag_morning and flag_night):
            df[col_names_day['feature'].iloc[pos_morning[0]] + '.' + 'morning' + '-' + 'night'] = df[col_names_day['col_names'].iloc[pos_morning[0]]] - df[col_names_day['col_names'].iloc[pos_night[0]]]
        if (flag_afternoon > 0 and flag_evening):
            df[col_names_day['feature'].iloc[pos_afternoon[0]] + '.' + 'afternoon' + '-' + 'evening'] = df[col_names_day['col_names'].iloc[pos_afternoon[0]]] - df[col_names_day['col_names'].iloc[pos_evening[0]]]
        if (flag_afternoon and flag_night):
            df[col_names_day['feature'].iloc[pos_afternoon[0]] + '.' + 'afternoon' + '-' + 'night'] = df[col_names_day['col_names'].iloc[pos_afternoon[0]]] - df[col_names_day['col_names'].iloc[pos_night[0]]]
        if (flag_evening and flag_night):
            df[col_names_day['feature'].iloc[pos_evening[0]] + '.' + 'evening' + '-' + 'night'] = df[col_names_day['col_names'].iloc[pos_evening[0]]] - df[col_names_day['col_names'].iloc[pos_night[0]]]

    return df


def intra_feature_extraction(df, feature_type):
    dfs = []
    df_sel_col_names = intra_feature(df)
    df = intra_feature_pairwise(df_sel_col_names, df)
    df_groups = df.groupby('participantID')
    df_group_names = df_groups.size().index
    for df_group_name in df_group_names:
        df_par = df_groups.get_group(df_group_name)
        if df_par.empty:
            # dfs.append(df_par)
            print('Warning: empty')
        else:
            participantID = df_par['participantID'].iloc[0]
            if feature_type == 'mean':
                df_par_features = df_par.mean(axis = 0)
            if feature_type == 'max':
                df_par_features = df_par.max(axis = 0)
            if feature_type == 'min':
                df_par_features = df_par.min(axis = 0)
            if feature_type == 'std':
                df_par_features = df_par.std(axis = 0)

            df_par_col_names = df_par.columns
            df_par_list = []
            for i in range(len(df_par_features)):
                df_par_list.append(df_par_features[i])

            df_par_list = [df_par_list]
            df_par_df = pd.DataFrame(df_par_list, columns = list(df_par_col_names))
            df_par_df['participantID'] = participantID
            dfs.append(df_par_df)

    df_par_output = pd.concat(dfs)

    return df_par_output


def inter_feature_extraction(df, feature_type):
    dfs = []
    df_groups = df.groupby('participantID')
    df_group_names = df_groups.size().index
    for df_group_name in df_group_names:
        df_par = df_groups.get_group(df_group_name)
        df_par_diff = row_diff(df_par)
        if df_par_diff.empty:
            # dfs.append(df_par_diff)
            print('Warning: empty')
        else:
            participantID = df_par_diff['participantID'].iloc[0]
            if feature_type == 'mean':
                df_par_diff_features = df_par_diff.mean(axis = 0)
            if feature_type == 'max':
                df_par_diff_features = df_par_diff.max(axis = 0)
            if feature_type == 'min':
                df_par_diff_features = df_par_diff.min(axis = 0)
            if feature_type == 'std':
                df_par_diff_features = df_par_diff.std(axis = 0)

            df_par_diff_col_names = df_par_diff.columns
            df_par_diff_list = []
            for i in range(len(df_par_diff_features)):
                df_par_diff_list.append(df_par_diff_features[i])

            df_par_diff_list = [df_par_diff_list]
            df_par_diff_df = pd.DataFrame(df_par_diff_list, columns = list(df_par_diff_col_names))
            df_par_diff_df['participantID'] = participantID
            dfs.append(df_par_diff_df)

    df_par_diff_output = pd.concat(dfs)

    return df_par_diff_output


def feature_extraction(df):
    df_intra_mean = intra_feature_extraction(df, 'mean')
    df_intra_max = intra_feature_extraction(df, 'max')
    df_intra_min = intra_feature_extraction(df, 'min')
    df_intra_std = intra_feature_extraction(df, 'std')
    df_inter_mean = inter_feature_extraction(df, 'mean')
    df_inter_max = inter_feature_extraction(df, 'max')
    df_inter_min = inter_feature_extraction(df, 'min')
    df_inter_std = inter_feature_extraction(df, 'std')

    return df_intra_mean, df_intra_max, df_intra_min, df_intra_std, df_inter_mean, df_inter_max, df_inter_min, df_inter_std