from pre_process import pre_proc
from pre_process import make_dir
from mi_imputation import impute_call
from mi_imputation import impute_iter
from feature_extraction import feature_extraction
from ml_models import agg_feats
from ml_models import add_labels
from ml_models import norm_data
from ml_models import lr_model
from ml_models import rf_model
from ml_models import xgb_model
from agg_results import agg_results
from agg_results import agg_results_lr
import os

if __name__ == '__main__':
    data_path = 'C:\\Users\\yanru\\OneDrive\\aapecs\\data_9.21'
    result_path = 'C:\\Users\\yanru\\OneDrive\\aapecs\\data_9.21\\results'
    file_name = '9-8_aware_personality.csv'
    pers_names = ['neoConscientiousness', 'neoNeuroticism', 'neoExtraversion', 'neoAgreeableness', 'neoOpenness', 'NegativeAffectivity', 'Detachment', 'Antagonism', 'Disinhibition', 'Psychoticism']
    thres = 0.3
    # Number of iteration in mice imputation
    iter = 5
    df_aware = pre_proc(data_path, file_name, 7)
    df_aware = impute_call(df_aware)
    col_del = ['Unnamed: 0', 'participantID', 'localDate']
    col_aware = df_aware.columns.tolist()
    for col_name in col_del:
        col_aware.remove(col_name)

    df_aware_new = df_aware[col_aware]
    for i in range(iter):
        print('Iteration:', i)
        df_aware_impute = impute_iter(df_aware_new, i)
        df_aware[col_aware] = df_aware_impute
        df_aware.drop(['Unnamed: 0', 'localDate'], axis = 1, inplace = True)
        df_intra_mean, df_intra_max, df_intra_min, df_intra_std, df_inter_mean, df_inter_max, df_inter_min, df_inter_std = feature_extraction(df_aware)
        df_agg = agg_feats(df_intra_mean, df_intra_max, df_intra_min, df_intra_std, df_inter_mean, df_inter_max, df_inter_min, df_inter_std)
        df_agg_new = add_labels(df_aware, df_agg, pers_names)
        df_norm, col_names = norm_data(df_agg, df_agg_new, pers_names)
        iter_path = result_path + '\\' + str(iter)
        os.chdir(make_dir(iter_path))
        print('Running lr model')
        df_lr_results = lr_model(df_norm, pers_names, col_names, thres)
        print('Running rf model')
        df_rf_results = rf_model(df_norm, pers_names,  col_names)
        print('Running xgb model')
        df_xgb_results = xgb_model(df_norm, pers_names, col_names)
        df_lr_results.to_csv('lr_results.csv', index=False)
        df_rf_results.to_csv('rf_results.csv', index=False)
        df_xgb_results.to_csv('xgb_results.csv', index=False)
        df_lr_results_agg = agg_results_lr(df_lr_results, 'personality')
        df_rf_results_agg = agg_results(df_rf_results, 'personality')
        df_xgb_results_agg = agg_results(df_xgb_results, 'personality')

    os.chdir(data_path)
    df_lr_results_agg.to_csv('lr_results_agg.csv')
    df_rf_results_agg.to_csv('rf_results_agg.csv')
    df_xgb_results_agg.to_csv('xgb_results_agg.csv')
