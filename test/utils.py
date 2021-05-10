import pandas as pd
from sklearn.preprocessing import LabelEncoder

from mtrec.functions.feature_column import *


def build_census(train_file, test_file, embed_dim=8):
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']
    train_df = pd.read_csv(train_file, header=None, names=column_names)
    test_df = pd.read_csv(test_file, header=None, names=column_names)
    # random
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    # concat
    data_df = pd.concat([train_df, test_df], axis=0)
    # First group of tasks according to the paper
    label_columns = ['income_50k', 'marital_stat']
    marital_mapping = {' Never married': 1, ' Married-civilian spouse present': 0,
                       ' Widowed': 0, ' Divorced': 0, ' Married-spouse absent': 0,
                       ' Separated': 0, ' Married-A F spouse present': 0}
    income_mapping = {' - 50000.': 0, ' 50000+.': 1}
    data_df['marital_stat'] = data_df['marital_stat'].map(marital_mapping)
    data_df['income_50k'] = data_df['income_50k'].map(income_mapping)
    data_y = data_df[label_columns]
    data_df = data_df.drop(columns=label_columns)
    data_y.columns = ['income', 'marital']
    label_columns = data_y.columns
    # Feature
    sparse_features = data_df.columns
    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    sparse_feature_columns = [sparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim)
     for feat in sparse_features]

    train_X = {name: data_df[name].iloc[:len(train_df)].values.astype('int32') for name in sparse_features}
    test_X = {name: data_df[name].iloc[len(train_df):].values.astype('int32') for name in sparse_features}
    train_y = {name: data_y[name].iloc[:len(train_df)].values.astype('int32') for name in label_columns}
    test_y = {name: data_y[name].iloc[len(train_df):].values.astype('int32') for name in label_columns}

    return sparse_feature_columns, (train_X, train_y), (test_X, test_y)


# train_file = 'data/census/census-income.data.gz'
# test_file = 'data/census/census-income.test.gz'
# build_census(train_file, test_file)