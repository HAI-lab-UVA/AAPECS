import pandas as pd
import numpy as np
import os
import math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost
import random


def mod_col(df, influx):
    col_names = df.columns
    col_names_new = []
    for col_name in col_names:
        col_names_new.append(influx + '_' + col_name)

    df.columns = col_names_new

    return df


def add_pers(index, df_aware, pers_names):
    aware_pers_row = []
    for pers_name in pers_names:
        aware_pers_vals = list(df_aware[df_aware['participantID'] == index][pers_name])
        if len(aware_pers_vals) > 0:
            aware_pers_row.append(aware_pers_vals[0])
        else:
            aware_pers_row.append(np.nan)

    return aware_pers_row


def sel_feats_corr(df_agg_cor, df_agg_cols, per_name, thres):
    unsel_feats = []
    for col_name in df_agg_cols:
        for row_name in df_agg_cols:
            if col_name == row_name:
                break
            else:
                cor_value = df_agg_cor.loc[col_name, row_name]
                if abs(cor_value) > thres:
                    if col_name not in unsel_feats and row_name not in unsel_feats:
                        col_per = df_agg_cor.loc[per_name, col_name]
                        row_per = df_agg_cor.loc[per_name, row_name]
                        if col_per < row_per:
                            unsel_feats.append(col_name)
#                             print(col_name)
                        else:
                            unsel_feats.append(row_name)
#                             print(row_name)

    sel_feats = []
    for feat in df_agg_cols:
        if feat not in unsel_feats:
            sel_feats.append(feat)

    return sel_feats


def feat_sel(model, feature_list):
    importances = list(model.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    feats = []
    feats_imp = []
    for feature_importance in feature_importances:
        if feature_importance[1] > 0:
            feats.append(feature_importance[0])
            feats_imp.append(feature_importance[1])

    return feats, feats_imp


def agg_feats(df_max, df_mean, df_min, df_std, df_diff_max, df_diff_mean, df_diff_min, df_diff_std):
    df_max.set_index(['participantID'], inplace=True)
    df_max = mod_col(df_max, 'intra_max')
    df_mean.set_index(['participantID'], inplace=True)
    df_mean = mod_col(df_mean, 'intra_mean')
    df_min.set_index(['participantID'], inplace=True)
    df_min = mod_col(df_min, 'intra_min')
    df_std.set_index(['participantID'], inplace=True)
    df_std = mod_col(df_std, 'intra_std')
    df_diff_max.set_index(['participantID'], inplace=True)
    df_diff_max = mod_col(df_diff_max, 'inter_max')
    df_diff_mean.set_index(['participantID'], inplace=True)
    df_diff_mean = mod_col(df_diff_mean, 'inter_mean')
    df_diff_min.set_index(['participantID'], inplace=True)
    df_diff_min = mod_col(df_diff_min, 'inter_min')
    df_diff_std.set_index(['participantID'], inplace=True)
    df_diff_std = mod_col(df_diff_std, 'inter_std')
    df_agg = pd.concat([df_max, df_mean, df_min, df_std, df_diff_max, df_diff_mean, df_diff_min, df_diff_std], axis = 1)
    # df_agg_cols = list(df_agg.columns)

    return  df_agg

def add_labels(df_aware, df_agg, pers_names):
    # pers_names = ['neoConscientiousness', 'neoNeuroticism', 'neoExtraversion', 'neoAgreeableness', 'neoOpenness', 'NegativeAffectivity', 'Detachment', 'Antagonism', 'Disinhibition', 'Psychoticism']
    df_agg_indexs = df_agg.index
    aware_pers = []
    for df_agg_index in df_agg_indexs:
        aware_pers.append(add_pers(df_agg_index, df_aware, pers_names))

    df_aware_pers = pd.DataFrame(aware_pers, columns = pers_names, index = df_agg_indexs)
    df_agg_new = pd.concat([df_agg, df_aware_pers], axis = 1)

    return df_agg_new


def norm_data(df_agg, df_agg_new, pers_names):
    df_agg_cols = list(df_agg.columns)
    sc = StandardScaler()
    agg_norm = sc.fit_transform(np.array(df_agg))
    df_agg_norm = pd.DataFrame(agg_norm.tolist(), columns = df_agg_cols)
    df_agg_norm.index = df_agg.index
    df_agg_new[df_agg_cols] = df_agg_norm[df_agg_cols]
    df_norm = df_agg_new.T.drop_duplicates().T
    col_names = list(df_agg.columns)

    return df_norm, col_names


def lr_model(df_agg_new, pers_names, col_names, thres):
    results = []
    model = LinearRegression()
    for pers_name in pers_names:
        # sel_feats = sel_features(df_agg_cor, col_names, pers_name, thres)
        parti_ids = df_agg_new.index
        train_set = df_agg_new
        test_set = df_agg_new[pers_name]
        for parti_id in parti_ids:
            test_id = [parti_id]
            test_set = df_agg_new.loc[test_id,:]
            train_ids = list(parti_ids)
            train_ids.remove(parti_id)
            train_set = df_agg_new.loc[train_ids,:]
            train_set_cor = train_set.corr()
            sel_feats = sel_feats_corr(train_set_cor, col_names, pers_name, thres)
            print('output')
            train_set_cor.to_csv('corr_test.csv')
            X_train = train_set[sel_feats]
            Y_train = train_set[pers_name]
            X_test = test_set[sel_feats]
            Y_test = test_set[pers_name]
            model.fit(X_train, Y_train)
            inter = model.intercept_
            coef = model.coef_
            Y_pred = model.predict(X_test)
            abs_error = abs(list((Y_pred - Y_test))[0])
            mse = list((Y_pred - Y_test) ** 2)[0]
            num = len(Y_test)
            mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / num * 100
            results.append([pers_name, parti_id, sel_feats, coef, inter, abs_error, mse, mape])
            df_results = pd.DataFrame(results, columns = ['personality', 'cl_paricipant_id', 'selected_features', 'coefficents', 'intercept', 'abs_error', 'mse', 'mape'])

    return df_results


def rf_model(df_agg_new, pers_names, col_names):
    target_pers = pers_names
    results = []
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    for pers_name in target_pers:
        parti_ids = df_agg_new.index
        train_set = df_agg_new
        test_set = df_agg_new[pers_name]
        for parti_id in parti_ids:
            test_id = [parti_id]
            test_set = df_agg_new.loc[test_id,:]
            train_ids = list(parti_ids)
            train_ids.remove(parti_id)
            train_set = df_agg_new.loc[train_ids,:]
            X_train = train_set[col_names]
            Y_train = train_set[pers_name]
            X_test = test_set[col_names]
            Y_test = test_set[pers_name]
            rf.fit(X_train, Y_train)
            Y_pred = rf.predict(X_test)
            feature_list = col_names
            abs_error = abs(list((Y_pred - Y_test))[0])
            mse = list((Y_pred - Y_test) ** 2)[0]
            sel_feats, feats_imp = feat_sel(rf, feature_list)
            num = len(Y_test)
            mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / num * 100
            results.append([pers_name, parti_id, sel_feats, feats_imp, abs_error, mse, mape])
            df_results = pd.DataFrame(results, columns = ['personality', 'cl_paricipant_id', 'selected_features', 'feature_importance', 'abs_error', 'mse', 'mape'])

    return results


def xgb_model(df_agg_new, pers_names, col_names):
    target_pers = pers_names
    results = []
    xgb = xgboost.XGBRegressor(n_estimators = 100, learning_rate = 0.01, max_depth = 5, seed = 8888)
    for pers_name in target_pers:
        parti_ids = df_agg_new.index
        train_set = df_agg_new
        test_set = df_agg_new[pers_name]
        for parti_id in parti_ids:
            test_id = [parti_id]
            test_set = df_agg_new.loc[test_id,:]
            train_ids = list(parti_ids)
            train_ids.remove(parti_id)
            train_set = df_agg_new.loc[train_ids,:]
        X_train = train_set[col_names]
        Y_train = train_set[pers_name]
        X_test = test_set[col_names]
        Y_test = test_set[pers_name]
        xgb.fit(X_train, Y_train)
        Y_pred = xgb.predict(X_test)
        feature_list = col_names
        abs_error = abs(list((Y_pred - Y_test))[0])
        mse = list((Y_pred - Y_test) ** 2)[0]
        sel_feats, feats_imp = feat_sel(xgb, feature_list)
        num = len(Y_test)
        mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / num * 100
        #print(mape)
        results.append([pers_name, parti_id, sel_feats, feats_imp, abs_error, mse, mape])

    return  results