from feature_extraction import feature_extraction
import os
import pandas as pd
from ml_models import agg_feats
from ml_models import add_labels


if __name__ == '__main__':
    data_path = 'C:\\Users\\yanru\\OneDrive\\aapecs\\data_9.21'
    pers_names = ['neoConscientiousness', 'neoNeuroticism', 'neoExtraversion', 'neoAgreeableness', 'neoOpenness', 'NegativeAffectivity', 'Detachment', 'Antagonism', 'Disinhibition', 'Psychoticism']
    os.chdir(data_path)
    df_test = pd.read_csv('imputed.csv')
    df_test.drop(['Unnamed: 0', 'localDate'], axis = 1, inplace = True)
    df_intra_mean, df_intra_max, df_intra_min, df_intra_std, df_inter_mean, df_inter_max, df_inter_min, df_inter_std = feature_extraction(df_test)
    df_agg = agg_feats(df_intra_mean, df_intra_max, df_intra_min, df_intra_std, df_inter_mean, df_inter_max, df_inter_min, df_inter_std)
    df_agg_new = add_labels(df_test, df_agg, pers_names)
    print(df_agg_new)
