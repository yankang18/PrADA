import pandas as pd
import numpy as np
from sklearn.utils import shuffle

COLUMNS_OF_HOST = ['age',
                   'education_year',
                   'capital_gain',
                   'capital_loss']

COLUMNS_OF_DEGREE_GUEST = ['age_bucket',  # 4
                           'marital_status',  # 5
                           'gender',  # 6
                           'native_country',  # 7
                           'race',  # 8
                           'workclass',  # 9
                           'occupation',  # 10
                           'education',  # 11
                           # "hours_per_week",      # 12
                           "relationship",  # 13
                           "income_label"]

COLUMNS_OF_ASIA_GUEST = ['age_bucket',  # 4
                         'marital_status',  # 5
                         'gender',  # 6
                         'native_country',  # 7
                         'race',  # 8
                         'workclass',  # 9
                         'occupation',  # 10
                         'education',  # 11
                         # "hours_per_week",      # 12
                         # "relationship",  # 13
                         "income_label"]

COLUMNS_OF_PPD_HOST = ['UserInfo_10',
                       'UserInfo_14',
                       'UserInfo_15',
                       'UserInfo_18']

COLUMNS_OF_PPD_GUEST = ['UserInfo_13',
                        'UserInfo_12',
                        'UserInfo_22',
                        'UserInfo_17',
                        'UserInfo_21',
                        'UserInfo_5',
                        'UserInfo_3',
                        'UserInfo_11',
                        'UserInfo_16',
                        'UserInfo_1',
                        'UserInfo_6',
                        'UserInfo_23',
                        'UserInfo_9',
                        'UserInfo_20_Longitude',
                        'UserInfo_20_Latitude',
                        'UserInfo_20_LongpLat',
                        'UserInfo_20_LongmLat',
                        'UserInfo_20_CityRank',
                        'UserInfo_CityRank_median',
                        'UserInfo_Latitude_median',
                        'UserInfo_Longitude_median',
                        'UserInfo_LongmLat_median',
                        'UserInfo_LongpLat_median',
                        'UserInfo_CityRank_std',
                        'UserInfo_Latitude_std',
                        'UserInfo_Longitude_std',
                        'UserInfo_LongmLat_std',
                        'UserInfo_LongpLat_std',
                        'ThirdParty_Info_1_median',
                        'ThirdParty_Info_10_median',
                        'ThirdParty_Info_11_median',
                        'ThirdParty_Info_12_median',
                        'ThirdParty_Info_13_median',
                        'ThirdParty_Info_14_median',
                        'ThirdParty_Info_15_median',
                        'ThirdParty_Info_16_median',
                        'ThirdParty_Info_17_median',
                        'ThirdParty_Info_2_median',
                        'ThirdParty_Info_3_median',
                        'ThirdParty_Info_4_median',
                        'ThirdParty_Info_5_median',
                        'ThirdParty_Info_6_median',
                        'ThirdParty_Info_7_median',
                        'ThirdParty_Info_8_median',
                        'ThirdParty_Info_9_median',
                        'ThirdParty_Info_1_std',
                        'ThirdParty_Info_10_std',
                        'ThirdParty_Info_11_std',
                        'ThirdParty_Info_12_std',
                        'ThirdParty_Info_13_std',
                        'ThirdParty_Info_14_std',
                        'ThirdParty_Info_15_std',
                        'ThirdParty_Info_16_std',
                        'ThirdParty_Info_17_std',
                        'ThirdParty_Info_2_std',
                        'ThirdParty_Info_3_std',
                        'ThirdParty_Info_4_std',
                        'ThirdParty_Info_5_std',
                        'ThirdParty_Info_6_std',
                        'ThirdParty_Info_7_std',
                        'ThirdParty_Info_8_std',
                        'ThirdParty_Info_9_std',
                        'ThirdParty_Info_1_min',
                        'ThirdParty_Info_10_min',
                        'ThirdParty_Info_11_min',
                        'ThirdParty_Info_12_min',
                        'ThirdParty_Info_13_min',
                        'ThirdParty_Info_14_min',
                        'ThirdParty_Info_15_min',
                        'ThirdParty_Info_16_min',
                        'ThirdParty_Info_17_min',
                        'ThirdParty_Info_2_min',
                        'ThirdParty_Info_3_min',
                        'ThirdParty_Info_4_min',
                        'ThirdParty_Info_5_min',
                        'ThirdParty_Info_6_min',
                        'ThirdParty_Info_7_min',
                        'ThirdParty_Info_8_min',
                        'ThirdParty_Info_9_min',
                        'ThirdParty_Info_1_max',
                        'ThirdParty_Info_10_max',
                        'ThirdParty_Info_11_max',
                        'ThirdParty_Info_12_max',
                        'ThirdParty_Info_13_max',
                        'ThirdParty_Info_14_max',
                        'ThirdParty_Info_15_max',
                        'ThirdParty_Info_16_max',
                        'ThirdParty_Info_17_max',
                        'ThirdParty_Info_2_max',
                        'ThirdParty_Info_3_max',
                        'ThirdParty_Info_4_max',
                        'ThirdParty_Info_5_max',
                        'ThirdParty_Info_6_max',
                        'ThirdParty_Info_7_max',
                        'ThirdParty_Info_8_max',
                        'ThirdParty_Info_9_max',
                        'ThirdParty_Info_1_first',
                        'ThirdParty_Info_10_first',
                        'ThirdParty_Info_11_first',
                        'ThirdParty_Info_12_first',
                        'ThirdParty_Info_13_first',
                        'ThirdParty_Info_14_first',
                        'ThirdParty_Info_15_first',
                        'ThirdParty_Info_16_first',
                        'ThirdParty_Info_17_first',
                        'ThirdParty_Info_2_first',
                        'ThirdParty_Info_3_first',
                        'ThirdParty_Info_4_first',
                        'ThirdParty_Info_5_first',
                        'ThirdParty_Info_6_first',
                        'ThirdParty_Info_7_first',
                        'ThirdParty_Info_8_first',
                        'ThirdParty_Info_9_first',
                        'Education_Info3',
                        'Education_Info5',
                        'Education_Info1',
                        'Education_Info7',
                        'Education_Info6',
                        'Education_Info8',
                        'Education_Info2',
                        'Education_Info4',
                        'SocialNetwork_9',
                        'SocialNetwork_6',
                        'SocialNetwork_14',
                        'SocialNetwork_8',
                        'SocialNetwork_10',
                        'SocialNetwork_17',
                        'SocialNetwork_16',
                        'SocialNetwork_13',
                        'SocialNetwork_3',
                        'SocialNetwork_4',
                        'SocialNetwork_15',
                        'SocialNetwork_5',
                        'SocialNetwork_2',
                        'SocialNetwork_7',
                        'SocialNetwork_12',
                        'WeblogInfo_57',
                        'WeblogInfo_30',
                        'WeblogInfo_26',
                        'WeblogInfo_17',
                        'WeblogInfo_6',
                        'WeblogInfo_28',
                        'WeblogInfo_4',
                        'WeblogInfo_3',
                        'WeblogInfo_14',
                        'WeblogInfo_35',
                        'WeblogInfo_15',
                        'WeblogInfo_34',
                        'WeblogInfo_33',
                        'WeblogInfo_8',
                        'WeblogInfo_27',
                        'WeblogInfo_7',
                        'WeblogInfo_18',
                        'WeblogInfo_36',
                        'WeblogInfo_29',
                        'WeblogInfo_39',
                        'WeblogInfo_48',
                        'WeblogInfo_9',
                        'WeblogInfo_38',
                        'WeblogInfo_25',
                        'WeblogInfo_16',
                        'WeblogInfo_24',
                        'WeblogInfo_5',
                        'WeblogInfo_56',
                        'WeblogInfo_42',
                        'WeblogInfo_2',
                        'WeblogInfo_19',
                        'WeblogInfo_21',
                        'WeblogInfo_20',
                        'target']

CENSUS_label = "income_label"
PPD_label = 'target'
FATE_label_name = 'y'


def load_pdd_data(data_dir):
    source_train_file_name = data_dir + "PPD_2014_1to9_train.csv"
    target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
    source_test_file_name = data_dir + 'PPD_2014_1to9_test.csv'
    target_test_file_name = data_dir + 'PPD_2014_10to12_test.csv'
    source_train = pd.read_csv(source_train_file_name, skipinitialspace=True)
    target_train = pd.read_csv(target_train_file_name, skipinitialspace=True)
    source_test = pd.read_csv(source_test_file_name, skipinitialspace=True)
    target_test = pd.read_csv(target_test_file_name, skipinitialspace=True)

    source_train['target'] = pd.to_numeric(source_train['target'], downcast='integer')
    target_train['target'] = pd.to_numeric(target_train['target'], downcast='integer')
    source_test['target'] = pd.to_numeric(source_test['target'], downcast='integer')
    target_test['target'] = pd.to_numeric(target_test['target'], downcast='integer')
    print(source_train.shape, target_train.shape, source_test.shape, target_test.shape)
    print(source_train.columns)
    print(target_test.head(10))

    return source_train, target_train, source_test, target_test


def load_census_asia_data(data_dir):
    source_train_file_name = data_dir + 'adult_source_train.csv'
    target_train_file_name = data_dir + 'adult_target_train.csv'
    source_test_file_name = data_dir + 'adult_source_test.csv'
    target_test_file_name = data_dir + 'adult_target_test.csv'

    source_train = pd.read_csv(source_train_file_name, skipinitialspace=True)
    target_train = pd.read_csv(target_train_file_name, skipinitialspace=True)
    source_test = pd.read_csv(source_test_file_name, skipinitialspace=True)
    target_test = pd.read_csv(target_test_file_name, skipinitialspace=True)

    print(source_train.shape, target_train.shape, source_test.shape, target_test.shape)
    print(source_train.columns)

    return source_train, target_train, source_test, target_test


def load_census_degree_data(data_dir):
    source_train_file_name = data_dir + 'degree_source_train.csv'
    target_train_file_name = data_dir + 'degree_target_train.csv'
    source_test_file_name = data_dir + 'degree_source_test.csv'
    target_test_file_name = data_dir + 'degree_target_test.csv'

    source_train = pd.read_csv(source_train_file_name, skipinitialspace=True)
    target_train = pd.read_csv(target_train_file_name, skipinitialspace=True)
    source_test = pd.read_csv(source_test_file_name, skipinitialspace=True)
    target_test = pd.read_csv(target_test_file_name, skipinitialspace=True)

    print(source_train.shape, target_train.shape, source_test.shape, target_test.shape)
    print(source_train.columns)

    return source_train, target_train, source_test, target_test


def prepare_data(source_train, target_train, source_test, target_test, columns_of_host, columns_of_guest):
    columns_of_guest = columns_of_guest
    columns = columns_of_host + columns_of_guest
    print(f"Host Columns:{columns_of_host}")
    print(f"Guest Columns:{columns_of_guest}")
    print(f"COLUMNS:{columns}")

    source_train = source_train[columns]
    target_train = target_train[columns]
    source_test = source_test[columns]
    target_test = target_test[columns]

    print(f"source_train shape:{source_train.shape}")
    print(f"target_train shape:{target_train.shape}")
    print(f"source_test shape:{source_test.shape}")
    print(f"target_test shape:{target_test.shape}")

    all_domains_train = pd.concat([source_train, target_train], axis=0)
    return all_domains_train, target_train, target_test


def prepare_fed_data(df_all_train, df_target_train, df_test, host_col_name_list, guest_col_name_list, task_tag,
                     label_name, save_dir):
    host_all_train_df = df_all_train[host_col_name_list]
    guest_all_train_df = df_all_train[guest_col_name_list].rename(columns={label_name: FATE_label_name})

    host_tgt_train_df = df_target_train[host_col_name_list]
    guest_tgt_train_df = df_target_train[guest_col_name_list].rename(columns={label_name: FATE_label_name})

    host_test_df = df_test[host_col_name_list]
    guest_test_df = df_test[guest_col_name_list].rename(columns={label_name: FATE_label_name})

    save_info = dict()
    save_info['host_df'] = host_all_train_df.reset_index(drop=True)
    save_info['guest_df'] = guest_all_train_df.reset_index(drop=True)
    save_info['domain'] = "all"
    save_info['data_mode'] = "train"
    save_info['task_tag'] = task_tag
    save_info['save_dir'] = save_dir
    save_fate_data(save_info)

    save_info['host_df'] = host_tgt_train_df.reset_index(drop=True)
    save_info['guest_df'] = guest_tgt_train_df.reset_index(drop=True)
    save_info['domain'] = "tgt"
    save_info['data_mode'] = "train"
    save_info['task_tag'] = task_tag
    save_info['save_dir'] = save_dir
    save_fate_data(save_info)

    save_info['host_df'] = host_test_df.reset_index(drop=True)
    save_info['guest_df'] = guest_test_df.reset_index(drop=True)
    save_info['domain'] = "tgt"
    save_info['data_mode'] = "test"
    save_info['task_tag'] = task_tag
    save_info['save_dir'] = save_dir
    save_fate_data(save_info)


def save_df(df, save_dir, save_file_name):
    full_file_name = save_dir + save_file_name
    df.to_csv(full_file_name, header=True, index=True, index_label="id")
    print(f"save {save_file_name} with shape {df.shape} to {full_file_name}.")


def save_fate_data(save_info):
    host_df = save_info['host_df']
    guest_df = save_info['guest_df']
    domain = save_info['domain']
    data_mode = save_info['data_mode']
    task_tag = save_info['task_tag']
    save_dir = save_info['save_dir']

    host_save_file_name = f"fate_{task_tag}_{domain}_host_{data_mode}_data.csv"
    guest_save_file_name = f"fate_{task_tag}_{domain}_guest_{data_mode}_data.csv"

    save_df(host_df, save_dir, host_save_file_name)
    save_df(guest_df, save_dir, guest_save_file_name)


if __name__ == "__main__":
    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    COLUMNS_OF_GUEST = COLUMNS_OF_PPD_GUEST
    COLUMNS_OF_HOST = COLUMNS_OF_PPD_HOST
    load_data = load_pdd_data
    degree_tag = "PPD"
    label_name = PPD_label

    # data_dir = "../datasets/census_processed/"
    # COLUMNS_OF_GUEST = COLUMNS_OF_DEGREE_GUEST
    # load_data = load_census_degree_data
    # degree_tag = "degree"

    # data_dir = "../datasets/census_processed/"
    # COLUMNS_OF_GUEST = COLUMNS_OF_ASIA_GUEST
    # load_data = load_census_asia_data
    # degree_tag = "asia"

    source_train, target_train, source_test, target_test = load_data(data_dir)

    all_domains_train, target_train, target_test = prepare_data(source_train, target_train,
                                                                source_test, target_test,
                                                                COLUMNS_OF_HOST, COLUMNS_OF_GUEST)

    save_dir = "/Users/yankang/Documents/Data/DANN/"
    prepare_fed_data(all_domains_train, target_train, target_test, COLUMNS_OF_HOST, COLUMNS_OF_GUEST, degree_tag,
                     label_name, save_dir)
