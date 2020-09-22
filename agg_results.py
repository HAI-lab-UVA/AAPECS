import pandas as pd
import numpy as np


def loc_index(data, target):
    re = 0
    for i in data:
        if i < target:
            break
        else:
            re += 1

    return re


def agg_results_lr(df, group_col):
    agg_results = []
    df_groups = df.groupby(group_col)
    df_groups_names = df_groups.size().index
    for df_groups_name in df_groups_names:
        df_group = df_groups.get_group(df_groups_name)
        agg_results_row = []
        feats = {}
        imp = {}
        abs_error = []
        mse = []
        mape = []
        for i in range(len(df_group)):
            if df_group['mape'].iloc[i] < 400:
                abs_error.append(df_group['abs_error'].iloc[i])
                mse.append(df_group['mse'].iloc[i])
                mape.append(df_group['mape'].iloc[i])
                selected_feats = df_group['selected_features'].iloc[i]
                selected_feat_comb = selected_feats[1 : (len(selected_feats) - 1)].split(',')
                for selected_feat in selected_feat_comb:
                    if feats.__contains__(selected_feat):
                        feats[selected_feat] = feats[selected_feat] + 1
                    else:
                        feats[selected_feat] = 1

        feat_by_freq = pd.DataFrame.from_dict(feats, orient = 'index', columns = ['freq'])
        feat_by_freq = feat_by_freq.reset_index().rename(columns = {'index' : 'feats'})
        feat_by_freq = feat_by_freq.sort_values(by = 'freq', ascending = False)

        freq_row = feat_by_freq['freq'].tolist()
        freq_row_med = 58
        freq_med_index = loc_index(freq_row, freq_row_med)

        agg_results_row = [df_groups_name, feat_by_freq['feats'].tolist()[0:freq_med_index], feat_by_freq['freq'].tolist()[0:freq_med_index], np.mean(abs_error), np.mean(mse), np.mean(mape)]
        agg_results.append(agg_results_row)

    df_agg_results = pd.DataFrame(agg_results, columns = ['personality', 'features_sort_by_freq', 'freq', 'abs_error', 'mse', 'mape'])
    return df_agg_results


def agg_results(df, group_col):
    agg_results = []
    df_groups = df.groupby(group_col)
    df_groups_names = df_groups.size().index
    for df_groups_name in df_groups_names:
        df_group = df_groups.get_group(df_groups_name)
        agg_results_row = []
        feats = {}
        imp = {}
        abs_error = []
        mse = []
        mape = []
        for i in range(len(df_group)):
            if df_group['mape'].iloc[i] < 400:
                abs_error.append(df_group['abs_error'].iloc[i])
                mse.append(df_group['mse'].iloc[i])
                mape.append(df_group['mape'].iloc[i])
                selected_feats = df_group['selected_features'].iloc[i]
                selected_feat_comb = selected_feats[1 : (len(selected_feats) - 1)].split(',')
                for selected_feat in selected_feat_comb:
                    if feats.__contains__(selected_feat):
                        feats[selected_feat] = feats[selected_feat] + 1
                    else:
                        feats[selected_feat] = 1

                feat_imp = df_group['feature_importance'].iloc[i]
                feat_imp_comb = feat_imp[1 : (len(feat_imp) - 1)].split(',')
                for j, selected_feat in enumerate(selected_feat_comb):
                    if imp.__contains__(selected_feat):
                        l_imp = imp[selected_feat]
                        l_imp.append(float(feat_imp_comb[j]))
                        imp[selected_feat] = l_imp
                    else:
                        imp[selected_feat] = [float(feat_imp_comb[j])]

        feat_by_freq = pd.DataFrame.from_dict(feats, orient = 'index', columns = ['freq'])
        feat_by_freq = feat_by_freq.reset_index().rename(columns = {'index' : 'feats'})
        feat_by_freq = feat_by_freq.sort_values(by = 'freq', ascending = False)

        for feat_imp in imp.keys():
            imp[feat_imp] = np.mean(imp[feat_imp])

        feat_by_imp = pd.DataFrame.from_dict(imp, orient = 'index', columns = ['imp'])
        feat_by_imp = feat_by_imp.reset_index().rename(columns = {'index': 'feats'})
        feat_by_imp = feat_by_imp.sort_values(by = 'imp', ascending = False)

        freq_row = feat_by_freq['freq'].tolist()
        imp_row = feat_by_imp['imp'].tolist()
        freq_row_med = 58
        imp_row_med = np.median(imp_row)
        freq_med_index = loc_index(freq_row, freq_row_med)
        imp_med_index = loc_index(imp_row, imp_row_med)

        agg_results_row = [df_groups_name, feat_by_freq['feats'].tolist()[0:freq_med_index], feat_by_freq['freq'].tolist()[0:freq_med_index], feat_by_imp['feats'].tolist()[0:imp_med_index], feat_by_imp['imp'].tolist()[0:imp_med_index], np.mean(abs_error), np.mean(mse), np.mean(mape)]
        agg_results.append(agg_results_row)

    df_agg_results = pd.DataFrame(agg_results, columns = ['personality', 'features_sort_by_freq', 'freq', 'features_sort_by_imp', 'imp','abs_error', 'mse', 'mape'])
    return df_agg_results