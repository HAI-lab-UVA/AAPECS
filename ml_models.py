import pandas as pd
import numpy as np
import copy
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import xgboost
from ml_param import param_space_lasso
from ml_param import param_space_lr
from ml_param import param_space_rf
from ml_param import param_space_xgb
from imputation import impute_iter
from pre_process import remove_missing
from param import num_gen
from feature_extraction import feature_extraction
from sklearn.model_selection import GridSearchCV
from param import pers_names, thres_sel_feat


def mod_col(df, influx):
    col_names = df.columns
    col_names_new = []
    for col_name in col_names:
        col_names_new.append(influx + '_' + col_name)

    df.columns = col_names_new

    return df


def add_pers(part_id, df_aware, pers_names):
    aware_pers_row = []
    for pers_name in pers_names:
        aware_pers_vals = list(df_aware[df_aware['participantID'] == part_id][pers_name])
        if len(aware_pers_vals) > 0:
            aware_pers_row.append(aware_pers_vals[0])
        else:
            print('No matching labels')

    return aware_pers_row

# def feat_sel(model, feature_list):
#     importances = list(model.feature_importances_)
#     feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
#     feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#     feats = []
#     feats_imp = []
#     for feature_importance in feature_importances:
#         if feature_importance[1] > 0:
#             feats.append(feature_importance[0])
#             feats_imp.append(feature_importance[1])
#
#     return feats, feats_imp


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
    parti_ids = df_agg['participantID']
    aware_pers = []
    for parti_id in parti_ids:
        aware_pers.append(add_pers(parti_id, df_aware, pers_names))

    df_aware_pers = pd.DataFrame(aware_pers, columns = pers_names, index = parti_ids)
    df_agg.index = parti_ids
    df_agg_new = pd.concat([df_agg, df_aware_pers], axis = 1)
    df_agg = df_agg.reset_index(drop=True)

    return df_agg_new


def lr_model():
    lr = LinearRegression()

    return lr


def lasso_model():
    lasso = Lasso()

    return lasso


def rf_model():
    rf = RandomForestRegressor()

    return rf


def xgb_model():
    xgb = xgboost.XGBRegressor()

    return xgb


def sel_feats_corr(df, pers_name, thres):
    df_corr = df.corr()
    df_col_names = df.columns.tolist()
    feat_names = copy.deepcopy(df_col_names)
    for i in pers_names:
        feat_names.remove(i)

    unsel_feats = []
    for col_name in feat_names:
        for row_name in feat_names:
            if col_name == row_name:
                break
            else:
                cor_value = df_corr.loc[col_name, row_name]
                if abs(cor_value) > thres:
                    if col_name not in unsel_feats and row_name not in unsel_feats:
                        col_per = df_corr.loc[pers_name, col_name]
                        row_per = df_corr.loc[pers_name, row_name]
                        if abs(col_per) < abs(row_per):
                            unsel_feats.append(col_name)
                        #                             print(col_name)
                        else:
                            unsel_feats.append(row_name)
    #                             print(row_name)

    sel_col_names = list()
    for j in df_col_names:
        if j not in unsel_feats and j not in pers_names:
            sel_col_names.append(j)

    return sel_col_names


def feat_imp(model, feature_list):
    importances = model.feature_importances_.tolist()
    feature_importances = [(feature, importance, 2) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    dict_feat_imp = dict()
    for feature_importance in feature_importances:
        if feature_importance[1] > 0:
            dict_feat_imp[feature_importance[0]] = feature_importance[1]

    return dict_feat_imp


def agg_feat_imp(ls_dicts):
    dict_agg_imp = dict()
    dict_agg_freq = dict()
    for dict_feat_imp in ls_dicts:
        for i in dict_feat_imp:
            if i in dict_agg_imp:
                dict_agg_imp[i] = dict_agg_imp[i] + dict_feat_imp[i]
            else:
                dict_agg_imp[i] = dict_feat_imp[i]

            if i in dict_agg_freq:
                dict_agg_freq[i] = dict_agg_freq[i] + 1
            else:
                dict_agg_freq[i] = 1

    for j in dict_agg_imp:
        dict_agg_imp[i] = dict_agg_imp[j] / dict_agg_freq[j]

    return dict_agg_imp, dict_agg_freq


def norm_data(df_train, df_test):
    df_train_feat = copy.deepcopy(df_train)
    df_test_feat = copy.deepcopy(df_test)
    df_train_feat = df_train_feat.drop(pers_names, axis=1)
    df_test_feat = df_test_feat.drop(pers_names, axis=1)
    train_feat_names = df_train_feat.columns.tolist()
    test_feat_names = df_test_feat.columns.tolist()
    train_participant_ids = df_train_feat.index.tolist()
    test_participant_ids = df_test_feat.index.tolist()
    sc = MinMaxScaler()
    sc.fit(np.array(df_train_feat))
    train_feat_norm = sc.transform(np.array(df_train_feat))
    test_feat_norm = sc.transform(np.array(df_test_feat))
    df_train_norm = pd.DataFrame(train_feat_norm.tolist(), index=train_participant_ids, columns=train_feat_names)
    df_test_norm = pd.DataFrame(test_feat_norm.tolist(), index=test_participant_ids, columns=test_feat_names)
    df_train[train_feat_names] = df_train_norm[train_feat_names]
    df_test[test_feat_names] = df_test_norm[test_feat_names]

    return df_train, df_test


def var_dis(df, avg_row):
    ls_results = list()
    for pers_name in pers_names:
        ls_traits = df[pers_name].tolist()
        val_mean = np.mean(ls_traits)
        ls_var = list()
        ls_dis = list()
        for i in ls_traits:
            ls_var.append(np.sqrt(i - val_mean))
            ls_dis.append(abs(i - val_mean))

        ls_results.append([pers_name, np.sum(ls_var)/(len(ls_traits) - avg_row), np.sum(ls_dis)/(len(ls_traits) - avg_row)])

    df_results = pd.DataFrame(ls_results, columns=['personality', 'variance', 'distance'])

    return df_results


def loocv_person(df, pers_name, ml_model_name):
    # target_pers = pers_names
    ls_results_loocv = list()
    ls_feat_loocv = list()
    parti_ids = df.index
    print('*****************************')
    print('Personality Trait:', pers_name)
    for iter_num, parti_id in enumerate(parti_ids):
        # ls_results_loocv = list()
        print('LOOCV ID:', iter_num)
        print('Participant ID: ', parti_id)
        test_id = [parti_id]
        test_set = df.loc[test_id, :]
        train_ids = copy.deepcopy(parti_ids.tolist())
        train_ids.remove(parti_id)
        train_set = df.loc[train_ids, :]
        # train_feat_set = feature_extraction(train_set)
        # test_feat_set = feature_extraction(test_set)
        col_names = train_set.columns.tolist()
        for i in pers_names:
            col_names.remove(i)
        # print('MICE Imputation: Train Set')
        # impute_train_sets = impute_iter(train_set, num_gen)
        # print('MICE Imputation: Test Set')
        # impute_test_sets = impute_iter(test_set, num_gen)
        # for i in range(num_gen):
        #     ls_results_mice = list()
        #     train_set_impute = impute_train_sets[i]
        #     test_set_impute = impute_test_sets[i]
        # train_set_norm = norm_data(train_set[feat_corr])
        # test_set_norm = norm_data(test_set[feat_corr])
        train_set_norm, test_set_norm = norm_data(train_set, test_set)
        feat_corr = sel_feats_corr(train_set_norm, pers_name, thres_sel_feat)
        print('Feature Selection:', len(feat_corr))
        X_train = train_set_norm[feat_corr]
        Y_train = train_set_norm[pers_name]
        X_test = test_set_norm[feat_corr]
        Y_test = test_set_norm[pers_name]
        print('Grid Search: ', ml_model_name)
        ml_model_best, ml_model_param_best = grid_search(X_train, Y_train, ml_model_name)
        Y_pred = ml_model_best.predict(X_test)
        # ml_model.fit(X_train, Y_train)
        # Y_pred = ml_model.predict(X_test)
        # abs_error = abs(list((Y_pred - Y_test))[0])
        # mse = list((Y_pred - Y_test) ** 2)[0]
        ls_results_iter = [pers_name, parti_id, Y_test.tolist(), Y_pred.tolist(), ml_model_param_best]
        if ml_model_name == 'xgb' or ml_model_name == 'rf':
            feature_list = col_names
            ls_feat_loocv.append(feat_imp(ml_model_best, feat_corr))

                # ls_results_iter.append(sel_feats)
                # ls_results_iter.append(feats_imp)
            # num = len(Y_test)
            # mape = sum(np.abs((Y_test - Y_pred) / Y_test)) / num * 100
            # abs_error, mse, mape = evaluate(ml_model_best, X_test, Y_test)
            # ls_results_mice.append([sel_feats, feats_imp, abs_error, mse, mape])
        # ls_results_mice.append(ls_results_iter)

        ls_results_loocv.append(ls_results_iter)
        # print(mape)
        # results.append([pers_name, parti_id, sel_feats, feats_imp, abs_error, mse, mape])

    # ls_results.append(ls_results_loocv)
    # df_results_loocv = pd.DataFrame(ls_results_loocv, columns = ['personality', 'paricipant_id', 'mice_iter', 'selected_features', 'feature_importance', 'Y_test', 'Y_pred'])
    # df_results = evaluate(df_results_loocv)

    return ls_results_loocv, ls_feat_loocv


def loocv_person_bench(df, pers_name):
    # target_pers = pers_names
    ls_results_loocv = list()
    parti_ids = df.index.tolist()
    for iter_num, parti_id in enumerate(parti_ids):
        test_id = [parti_id]
        test_set = df.loc[test_id, :]
        train_ids = copy.deepcopy(parti_ids)
        train_ids.remove(parti_id)
        train_set = df.loc[train_ids, :]
        Y_train = train_set[pers_name]
        Y_test = test_set[pers_name]
        Y_pred = np.mean(Y_train.tolist())
        ls_results_iter = [pers_name, parti_id, Y_test.tolist(), [Y_pred]]
        ls_results_loocv.append(ls_results_iter)

    return ls_results_loocv


def loocv_day(df, pers_name, ml_model_name):
    # target_pers = pers_names
    ls_results_loocv = list()
    ls_feat_loocv = list()
    parti_ids = list(set(df['participantID'].tolist()))
    print('*****************************')
    print('Personality Trait:', pers_name)
    for iter_num, parti_id in enumerate(parti_ids):
        print(parti_id)
        print('LOOCV ID:', iter_num)
        print('Participant ID: ', parti_id)
        test_id = [parti_id]
        test_set = df[df['participantID'].isin(test_id)]
        test_set.reset_index(drop=True, inplace=True)
        test_set.drop(['participantID'], axis=1)
        train_ids = copy.deepcopy(parti_ids)
        train_ids.remove(parti_id)
        train_set = df[df['participantID'].isin(train_ids)]
        train_set.reset_index(drop=True, inplace=True)
        train_set.drop(['participantID'], axis=1)
        col_names = train_set.columns.tolist()
        for i in pers_names:
            col_names.remove(i)

        print('MICE Imputation: Test Set')
        impute_test_sets, feat_names = impute_iter(test_set, col_names, num_gen)
        print('MICE Imputation: Train Set')
        impute_train_sets, feat_names = impute_iter(train_set, feat_names, num_gen)
        for j in range(num_gen):
            train_set_impute = impute_train_sets[j]
            test_set_impute = impute_test_sets[j]
            train_set_norm, test_set_norm = norm_data(train_set_impute, test_set_impute)
            feat_corr = sel_feats_corr(train_set_norm, pers_name, thres_sel_feat)
            print('Feature Selection:', len(feat_corr))
            X_train = train_set_norm[feat_corr]
            Y_train = train_set_norm[pers_name]
            X_test = test_set_norm[feat_corr]
            Y_test = test_set_norm[pers_name]
            ml_model = train_no_tuning(X_train, Y_train, ml_model_name)
            Y_pred = ml_model.predict(X_test)
            ls_results_iter = [pers_name, parti_id, j, Y_test.tolist(), Y_pred.tolist()]
            if ml_model_name == 'xgb' or ml_model_name == 'rf':
                feature_list = feat_names
                ls_feat_loocv.append(feat_imp(ml_model, feat_corr))

            ls_results_loocv.append(ls_results_iter)

    return ls_results_loocv, ls_feat_loocv


def loocv_day_bench(df, pers_name):
    # target_pers = pers_names
    parti_ids = list(set(df['participantID'].tolist()))
    ls_results_loocv = list()
    for iter_num, parti_id in enumerate(parti_ids):
        test_id = [parti_id]
        test_set = df[df['participantID'].isin(test_id)]
        test_set.reset_index(drop=True, inplace=True)
        test_set.drop(['participantID'], axis=1)
        train_ids = copy.deepcopy(parti_ids)
        train_ids.remove(parti_id)
        train_set = df[df['participantID'].isin(train_ids)]
        train_set.reset_index(drop=True, inplace=True)
        train_set.drop(['participantID'], axis=1)
        Y_train = train_set[pers_name]
        Y_test = list(set(test_set[pers_name].tolist()))
        Y_pred = np.mean(Y_train.tolist())
        ls_results_iter = [pers_name, parti_id, Y_test, [Y_pred]]
        ls_results_loocv.append(ls_results_iter)

    return ls_results_loocv


def combine_param(param_space, param_names, accum, param_combs):
    last = (len(param_names) == 1)
    ls_params = param_space[param_names[0]]
    n = len(ls_params)
    for i in range(n):
        accum[param_names[0]] = ls_params[i]
        if last:
            param_combs.append(copy.deepcopy(accum))
        else:
            param_combs = combine_param(param_space, param_names[1:], accum, param_combs)

    return param_combs


def MAE(predict, target):
    predict = np.array(predict)
    target = np.array(target)
    return (abs(predict - target)).mean()


def MSE(predict, target):
    predict = np.array(predict)
    target = np.array(target)
    return ((predict - target) ** 2).mean()


def RMSE(predict, target):
    predict = np.array(predict)
    target = np.array(target)
    return np.sqrt(((predict - target) ** 2).mean())


def MAPE(predict, target):
    ls_iter = list()
    for i in range(len(predict)):
        if target[i] != 0:
            ls_iter.append((abs(predict[i] - target[i])) / target[i])
        else:
            print(target[i])

    return np.mean(ls_iter) * 100


def R2(predict, target):
    return r2_score(target, predict)


def R_SQR(predict, target):
    r2 = R2(predict, target)

    return np.sqrt(r2)


def select_best_model(df_results, df_results_best, param_comb, param_comb_best, best_mse):
    cur_mse = df_results['mse'].mean()
    if cur_mse < best_mse:
        df_results_best = df_results
        param_comb_best = param_comb
    return df_results_best, param_comb_best


def train_no_tuning(X_train, Y_train, ml_model_name):
    if ml_model_name == 'lr':
        param_space = param_space_lr
        ml_model = lr_model()
    elif ml_model_name == 'lasso':
        param_space = param_space_lasso
        ml_model = lasso_model()
    elif ml_model_name == 'rf':
        param_space = param_space_rf
        ml_model = rf_model()
    elif ml_model_name == 'xgb':
        param_space = param_space_xgb
        ml_model = xgb_model()
    else:
        sys.exit('Error: Wrong Input Machine Learning Name')

    ml_model.fit(X_train, Y_train)

    return ml_model


def grid_search(X_train, Y_train, ml_model_name):
    if ml_model_name == 'lr':
        param_space = param_space_lr
        ml_model = lr_model()
    elif ml_model_name == 'lasso':
        param_space = param_space_lasso
        ml_model = lasso_model()
    elif ml_model_name == 'rf':
        param_space = param_space_rf
        ml_model = rf_model()
    elif ml_model_name == 'xgb':
        param_space = param_space_xgb
        ml_model = xgb_model()
    else:
        sys.exit('Error: Wrong Input Machine Learning Name')

    # param_names = list(param_space.keys())
    # param_combs = combine_param(param_space, param_names, accum, param_combs)
    grid_search = GridSearchCV(estimator=ml_model, param_grid=param_space, cv=5, verbose=1)
    grid_search.fit(X_train, Y_train)
    best_grid_model = grid_search.best_estimator_
    best_grid_model_param = grid_search.best_params_
    # grid_accuracy = evaluate(best_grid, X_test, Y_test)
    # for param_comb in param_combs:
    #     if ml_model_name == 'lr':
    #         ml_model = lr_model(param_comb)
    #     elif ml_model_name == 'lasso':
    #         ml_model = lasso_model(param_comb)
    #     elif ml_model_name == 'rf':
    #         ml_model = rf_model(param_comb)
    #     else:
    #         ml_model = xgb_model(param_comb)
    #
    #     df_results = loocv(df, pers_name, col_names, ml_model)
    #     df_results_best, param_comb_best = select_best_model(df_results, df_results_best, best_mse)

    return best_grid_model, best_grid_model_param


def agg_results(df_results):
    ls_evalutation = list()
    df_results_groups = df_results.groupby('Personality')
    df_results_group_names = df_results_groups.size().index
    for df_results_group_name in df_results_group_names:
        ls_evaluation_row = [df_results_group_name]
        df_results_group = df_results_groups.get_group(df_results_group_name)
        ls_test_vals = list()
        ls_pred_vals = list()
        for val in df_results_group['Y test'].tolist():
            if len(val) != 1:
                print('Error')

            ls_test_vals.append(val[0])

        for val in df_results_group['Y pred'].tolist():
            if len(val) != 1:
                print('Error')

            ls_pred_vals.append(val[0])

        df_evaluation_input = pd.DataFrame()
        df_evaluation_input['Y_test'] = ls_test_vals
        df_evaluation_input['Y_pred'] = ls_pred_vals

        #     df_evalution_output_xgb = evaluate(df_evaluation_input_xgb)
        ls_evalutation.append([df_results_group_name,
                               MAE(ls_pred_vals, ls_test_vals),
                               MSE(ls_pred_vals, ls_test_vals),
                               RMSE(ls_pred_vals, ls_test_vals),
                               MAPE(ls_pred_vals, ls_test_vals),
                               R2(ls_pred_vals, ls_test_vals)])

    df_results_summary = pd.DataFrame(ls_evalutation,
                                      columns=['Personality', 'MAE', 'MSE', 'RMSE', 'MAPE(%)', 'R2'])

    return df_results_summary






