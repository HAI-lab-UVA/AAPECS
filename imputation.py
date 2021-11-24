import numpy as np
import pandas as pd
import copy
import scipy.stats as st
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from param import pers_names


def mcar_test(data):
    dataset = data.copy()
    vars = dataset.dtypes.index.values
    n_var = dataset.shape[1]

    # mean and covariance estimates
    # ideally, this is done with a maximum likelihood estimator
    gmean = dataset.mean()
    gcov = dataset.cov()

    # set up missing data patterns
    r = 1 * dataset.isnull()
    mdp = np.dot(r, list(map(lambda x: ma.pow(2, x), range(n_var))))
    sorted_mdp = sorted(np.unique(mdp))
    n_pat = len(sorted_mdp)
    correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
    dataset['mdp'] = pd.Series(correct_mdp, index=dataset.index)

    # calculate statistic and df
    pj = 0
    d2 = 0
    for i in range(n_pat):
        dataset_temp = dataset.loc[dataset['mdp'] == i, vars]
        select_vars = ~dataset_temp.isnull().any()
        pj += np.sum(select_vars)
        select_vars = vars[select_vars]
        means = dataset_temp[select_vars].mean() - gmean[select_vars]
        select_cov = gcov.loc[select_vars, select_vars]
        mj = len(dataset_temp)
        parta = np.dot(means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[1])))
        d2 += mj * (np.dot(parta, means))

    df = pj - n_var

    # perform test and save output
    p_value = 1 - st.chi2.cdf(d2, df)

    return p_value


def fill_0(df, call_len, call_col, i):
    df_output = copy.deepcopy(df)
    for j in range(call_len):
        df_output[call_col[j]].iloc[i] = 0

    return df_output


# fill missing call data
def impute_call(df):
    call_col1 = []
    call_col2 = []
    call_col3 = []
    col_names = df.columns
    for col_name in col_names:
        if 'incoming' in col_name:
            call_col1.append(col_name)
        if 'outgoing' in col_name:
            call_col2.append(col_name)
        if 'missed' in col_name:
            call_col3.append(col_name)

    call_len1 = len(call_col1)
    call_len2 = len(call_col2)
    call_len3 = len(call_col3)

    for i in range(len(df)):
        call_row1 = df[call_col1].iloc[i]
        if call_row1.isna().sum() / call_len1 >= 1:
            df = fill_0(df, call_len1, call_col1, i)
        call_row2 = df[call_col2].iloc[i]
        if call_row2.isna().sum() / call_len2 >= 1:
            df = fill_0(df, call_len2, call_col2, i)
        call_row3 = df[call_col3].iloc[i]
        if call_row3.isna().sum() / call_len3 >= 1:
            df = fill_0(df, call_len3, call_col3, i)

    return df


# multiple imputation
def impute_iter(df, col_names, num):
    ls_df_impute = list()
    df_col = copy.deepcopy(df[col_names])
    df_feat = df_col[df_col.columns[~df_col.isnull().all()]]
    feat_names = df_feat.columns.tolist()
    for i in range(num):
        print('Imputation: ', i)
        imp = IterativeImputer(max_iter=1, random_state=i * 60, sample_posterior=True)
        imp.fit(df_feat)
        data_impute = imp.transform(df_feat)
        df_impute = pd.DataFrame(data_impute, columns=feat_names)
        # df_impute['participantID'] = ls_id
        col_names_output = copy.deepcopy(feat_names)
        col_names_output.extend(pers_names)
        df_output = copy.deepcopy(df)[col_names_output]
        df_output[feat_names] = df_impute
        ls_df_impute.append(df_output)

    return ls_df_impute, feat_names


# fill missing with min
def impute_min(df):
    ls_impute = list()
    ls_cols = df.columns.tolist()
    ls_index = df.index.tolist()
    ls_col_min = list()
    for m in ls_cols:
        ls_col = df[m].tolist()
        ls_col_na = list()
        for n in ls_col:
            if not pd.isna(n):
                ls_col_na.append(n)

        ls_col_min.append(np.min(ls_col_na))

    for i in ls_index:
        ls_row = df.loc[i,].tolist()
        ls_row_impute = list()
        for k, j in enumerate(ls_row):
            if pd.isna(j):
                ls_row_impute.append(ls_col_min[k] - 0.1)
            else:
                ls_row_impute.append(j)

        ls_impute.append(ls_row_impute)

    df_impute = pd.DataFrame(ls_impute, index=ls_index, columns=ls_cols)

    return df_impute


def agg_feats(feats_old, imp_old, feats_new, imp_new, num):
    feats_agg = []
    imp_agg = []
    imp_old = (pd.Series(imp_old) * (num - 1)).tolist()
    if (len(feats_old) > len(feats_new)):
        feats_l = copy.deepcopy(feats_old)
        imp_l = copy.deepcopy(imp_old)
        feats_s = copy.deepcopy(feats_new)
        imp_s = copy.deepcopy(imp_new)
    else:
        feats_l = copy.deepcopy(feats_new)
        imp_l = copy.deepcopy(imp_new)
        feats_s = copy.deepcopy(feats_old)
        imp_s = copy.deepcopy(imp_old)

    for i in range(len(feats_l)):
        feat_l = feats_l[i]
        imp_l_v = imp_l[i]
        try:
            p = feats_s.index(feat_l)
            feat_agg = feat_l
            imp_agg_n = (imp_s[p] + imp_l_v) / num
        except:
            imp_agg_n = imp_l_v / num

        feats_agg.append(feat_l)
        imp_agg.append(imp_agg_n)

    return feats_agg, imp_agg


def agg_metric(v_old, v_new):
    v_agg = (v_new - v_old) / 2 + v_old

    return v_agg


def agg_iter(df_old, df_new, num):
    df_agg = []
    if len(df_old) == len(df_new):
        personality_names = df_old['personality']
        paricipant_ids = df_old['cl_paricipant_id']
        for i in range(len(personality_names)):
            personality_name = personality_names[i]
            paricipant_id = paricipant_ids[i]
            df_com = df_new[(df_new['personality'] == personality_name) & (df_new['cl_paricipant_id'] == paricipant_id)]
            if len(df_com) == 1:
                feats_old = df_old.loc[i, 'selected_features']
                imp_old = df_old.loc[i, 'feature_importance']
                feats_new = df_com['selected_features'].tolist()[0]
                imp_new = df_com['feature_importance'].tolist()[0]
                feats_agg, imp_agg = agg_feats(feats_old, imp_old, feats_new, imp_new, num)
                abs_error_agg = agg_metric(df_old.loc[i, 'abs_error'], df_com.loc[i, 'abs_error'])
                mse_agg = agg_metric(df_old.loc[i, 'mse'], df_com.loc[i, 'mse'])
                mape_agg = agg_metric(df_old.loc[i, 'mape'], df_com.loc[i, 'mape'])
                df_agg_row = [personality_name, paricipant_id, feats_agg, imp_agg, abs_error_agg, mse_agg, mape_agg]

            df_agg.append(df_agg_row)

        df_agg = pd.DataFrame(df_agg, columns=df_old.columns.tolist())

    return df_agg
