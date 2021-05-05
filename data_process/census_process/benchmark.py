import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from data_process.benchmark_utils import run_benchmark
from data_process.census_process.mapping_resource import continuous_cols, categorical_cols, target_col_name

# COLUMNS = ['age',
#            'education_year',
#            'hours_per_week',
#            'capital_gain',
#            'capital_loss',
#            'age_bucket',
#            'marital_status',
#            'relationship',
#            'gender',
#            # 'native_country',
#            'race',
#            'workclass',
#            'occupation',
#            'education',
#            'is_asian',
#            "income_label"]

COLUMNS = ['age',
           'education_year',
           # 'hours_per_week',
           'capital_gain',
           'capital_loss',
           'age_bucket',
           'marital_status',
           # 'relationship',
           'gender',
           # 'native_country',
           'race',
           'workclass',
           'occupation',
           'education',
           # 'is_asian',
           "income_label"]


# COLUMNS_TO_LOAD = ['age',                 # 0
#                    'education_year',      # 1
#                    'capital_gain',        # 2
#                    'capital_loss',        # 3
#                    'age_bucket',          # 4
#                    'marital_status',      # 5
#                    'gender',              # 6
#                    'native_country',      # 7
#                    'race',                # 8
#                    'workclass',           # 9
#                    'occupation',          # 10
#                    'education',           # 11
#                    # "hours_per_week",      # 12
#                    # "relationship",        # 13
#                    "income_label"]


def train_benchmark(samples_train, samples_test):
    train_data, train_label = samples_train[:, :-1], samples_train[:, -1]
    test_data, test_label = samples_test[:, :-1], samples_test[:, -1]

    print(f"train_data shape : {train_data.shape}")
    print(f"train_label shape : {train_label.shape}")
    print(f"test_data shape : {test_data.shape}")
    print(f"test_label shape : {test_label.shape}")
    run_benchmark(train_data, train_label, test_data, test_label)

    # print(f"train_data shape : {train_data.shape}")
    # print(f"train_label shape : {train_label.shape}")
    # print(f"test_data shape : {test_data.shape}")
    # print(f"test_label shape : {test_label.shape}")
    # # lr_adult = LogisticRegressionCV(cv=5, random_state=0, max_iter=500)
    # lr_adult = LogisticRegression(max_iter=500)
    # # lr_adult = LogisticRegression(max_iter=500)
    # cls = lr_adult.fit(train_data, train_label)
    # pred = cls.predict(test_data)
    # test_label = test_label.astype(float)
    # # print("pred:", pred, pred.shape, type(pred), sum(pred))
    # # print("test_label:", test_label, test_label.shape, type(test_label), sum(test_label))
    # # pred_prob = cls.predict_proba(test_data)
    # # print("pred_prob shape:", pred_prob.shape)
    # acc = accuracy_score(pred, test_label)
    # auc = roc_auc_score(pred, test_label)
    # res = precision_recall_fscore_support(pred, test_label, average='macro')
    # print(f"accuracy : {acc}")
    # print(f"auc : {auc}")
    # print(f"prf : {res}")
    # print(f"coef: {cls.coef_}")


def train_census_adult():
    adult = pd.read_csv('../../datasets/census_processed/standardized_adult.csv', skipinitialspace=True)
    adult = adult[COLUMNS]

    adult_test = pd.read_csv('../../datasets/census_processed/standardized_adult_test.csv', skipinitialspace=True)
    adult_samples_test = adult_test[COLUMNS].values

    # adult_samples = adult.values
    # adult_samples_train, adult_samples_test = train_test_split(adult_samples, train_size=0.7)
    adult_samples_train = adult.values

    print("adult_samples_train", adult_samples_train.shape)
    print("adult_samples_test", adult_samples_test.shape)
    train_benchmark(adult_samples_train, adult_samples_test)


def prepare_census_95_train_data(num_sample=1000):
    census95 = pd.read_csv('../../datasets/census_processed/sampled_standardized_census95.csv', skipinitialspace=True)
    census95 = census95[COLUMNS]

    census95_1 = census95[census95["income_label"] == 1]
    census95_0 = census95[census95["income_label"] == 0]
    census95_1 = shuffle(census95_1)
    census95_0 = shuffle(census95_0)

    print(f"original census95 with label 1 has shape:{census95_1.shape}")
    print(f"original census95 with label 0 has shape:{census95_0.shape}")

    num_sample_1 = int(num_sample / 4)
    num_sample_0 = num_sample - num_sample_1
    census95_1_500 = census95_1.values[:num_sample_1]
    census95_0_500 = census95_0.values[:num_sample_0]
    census95_1000 = np.concatenate([census95_1_500, census95_0_500], axis=0)
    census95_1000 = shuffle(census95_1000)
    print(f"census95_1_500 has shape:{census95_1_500.shape}")
    print(f"census95_0_500 has shape:{census95_0_500.shape}")
    print(f"census95_1000 has shape:{census95_1000.shape}")

    census95_1000_df = pd.DataFrame(data=census95_1000, columns=COLUMNS)
    census95_1000_df.to_csv('../datasets/census_processed/standardized_census95_' + "train_" + str(num_sample) + ".csv",
                            index=False)
    return census95_1000_df


def train_census_95():
    # census95 = pd.read_csv('../datasets/census_processed/sampled_standardized_census95.csv', skipinitialspace=True)
    # census95 = pd.read_csv('../datasets/census_processed/standardized_census95_benchmark_train_9768.csv',
    #                        skipinitialspace=True)
    census95 = pd.read_csv('../../datasets/census_processed/sampled_standardized_census95_train.csv',
                           skipinitialspace=True)
    census95 = census95[COLUMNS]
    census95_samples_train = census95.values

    census95_test = pd.read_csv('../../datasets/census_processed/standardized_census95_test.csv',
                                skipinitialspace=True)
    # census95_test = sample_data(census95_test, num_samples=15000)
    census95_samples_test = census95_test[COLUMNS].values

    print(census95[census95["income_label"] == 1].shape)
    print(census95[census95["income_label"] == 0].shape)

    # census95_samples_train, _ = train_test_split(census95_samples_train, train_size=0.3, shuffle=True)
    # print("census95_samples_train", census95_samples_train.shape)
    # print("census95_samples_test", census95_samples_test.shape)
    # num_sample = census95_samples_train.shape[0]
    # census95_1000_df = pd.DataFrame(data=census95_samples_train, columns=COLUMNS)
    # print(census95_1000_df[census95_1000_df["income_label"] == 1].shape)
    # print(census95_1000_df[census95_1000_df["income_label"] == 0].shape)
    # census95_1000_df.to_csv(
    #     '../datasets/census_processed/standardized_census95_benchmark_' + "train_" + str(num_sample) + ".csv",
    #     index=False)

    train_benchmark(census95_samples_train, census95_samples_test)
    # train_benchmark(census95_1000, census95_samples_test)


def train_adult_to_95():
    adult = pd.read_csv('../../datasets/census_processed/standardized_adult.csv', skipinitialspace=True)
    adult = adult[COLUMNS]
    census95 = pd.read_csv('../../datasets/census_processed/sampled_standardized_census95.csv', skipinitialspace=True)
    census95 = census95[COLUMNS]

    adult_samples = adult.values
    census95_samples = census95.values

    census95_samples = shuffle(census95_samples)
    ratio_of_95_to_train = 0.3
    num_train = int(ratio_of_95_to_train * len(census95_samples))
    census95_samples_train = census95_samples[:num_train]

    census95_test = pd.read_csv('../../datasets/census_processed/sampled_standardized_census95_test.csv',
                                skipinitialspace=True)
    census95_samples_test = census95_test[COLUMNS].values
    # census95_samples_test = census95_samples[-5000:]

    print("census95_samples_train shape:", census95_samples_train.shape)

    comb_samples_train = np.concatenate([adult_samples, census95_samples_train], axis=0)
    comb_samples_train = shuffle(comb_samples_train)

    train_benchmark(comb_samples_train, census95_samples_test)


def train_on_dann(file_dict, columns=None):
    source_train_file = file_dict['source_train_file']
    target_train_file = file_dict['target_train_file']
    target_test_file = file_dict['target_test_file']
    data_dir = file_dict['data_dir']

    adult_source_train = pd.read_csv(data_dir + source_train_file, skipinitialspace=True)
    adult_target_train = pd.read_csv(data_dir + target_train_file, skipinitialspace=True)
    adult_target_test = pd.read_csv(data_dir + target_test_file, skipinitialspace=True)

    if columns:
        adult_source_train = adult_source_train[columns]
        adult_target_train = adult_target_train[columns]
        adult_target_test = adult_target_test[columns]

    print(f"source_train shape:{adult_source_train.shape}")
    print(f"target_train shape:{adult_target_train.shape}")
    print(f"target_test shape:{adult_target_test.shape}")

    print("====== train model only on target =======")
    adult_target_train = adult_target_train.values
    adult_target_test = adult_target_test.values
    adult_target_train = shuffle(adult_target_train)
    adult_target_test = shuffle(adult_target_test)
    train_benchmark(adult_target_train, adult_target_test)
    print('\n')
    print('\n')
    print("====== train model on src+tgt =======")
    adult_source_train = adult_source_train.values
    adult_all_train = np.concatenate([adult_source_train, adult_target_train], axis=0)
    adult_all_train = shuffle(adult_all_train)
    train_benchmark(adult_all_train, adult_target_test)


def train_adult():
    src_tgt_train = pd.read_csv('../../datasets/census_processed/standardized_adult_train.csv', skipinitialspace=True)
    src_tgt_train = src_tgt_train[COLUMNS]
    adult_test = pd.read_csv('../../datasets/census_processed/standardized_adult_test.csv', skipinitialspace=True)
    adult_test = adult_test[COLUMNS]

    target_train = src_tgt_train[src_tgt_train['is_asian'] == 1]
    source_train = src_tgt_train[src_tgt_train['is_asian'] == 0]
    print(f"target train shape:{target_train.shape}")
    print(f"source train shape:{source_train.shape}")

    target_test = adult_test[adult_test['is_asian'] == 1]
    source_test = adult_test[adult_test['is_asian'] == 0]
    print(f"target test shape:{target_test.shape}")
    print(f"source test shape:{source_test.shape}")

    print("====== train asia adult =======")
    target_train = target_train.values
    target_test = target_test.values
    target_train = shuffle(target_train)
    target_test = shuffle(target_test)
    train_benchmark(target_train, target_test)

    # print("====== train non-asia adult =======")
    # source_train = source_train.values
    # source_test = source_test.values
    # source_train = shuffle(source_train)
    # source_test = shuffle(source_test)
    # train_benchmark(source_train, source_test)

    print("====== train all for asia adult =======")
    src_tgt_train = np.concatenate([target_train, source_train], axis=0)
    src_tgt_train = shuffle(src_tgt_train)
    train_benchmark(src_tgt_train, target_test)


if __name__ == "__main__":

    # prepare_census_95_train_data(num_sample=1000)
    # train_census_adult()
    # train_census_95()
    # train_adult_to_95()
    # train_adult()

    # source_train_file = pd.read_csv('../datasets/census_processed/adult_source_train.csv', skipinitialspace=True)
    # source_train_file = adult_source_train[COLUMNS]
    # target_train_file = pd.read_csv('../datasets/census_processed/adult_target_train.csv', skipinitialspace=True)
    # target_train_file = adult_target_train[COLUMNS]
    # adult_target_test = pd.read_csv('../datasets/census_processed/adult_target_test.csv', skipinitialspace=True)
    # adult_target_test = adult_target_test[COLUMNS]

    # source_train_file = "adult_source_train.csv"
    # target_train_file = 'adult_target_train.csv'
    # target_test_file = 'adult_target_test.csv'
    # data_dir = '../datasets/census_processed/'
    # columns_list = COLUMNS

    # source_train_file = "PPD_2014_1to9_train.csv"
    # target_train_file = 'PPD_2014_10to12_train.csv'
    # # source_test_file = 'PPD_2014_1to9_test.csv'
    # target_test_file = 'PPD_2014_10to12_test.csv'
    # data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    # columns_list = None

    # source_train_file = 'degree_source_train.csv'
    # target_train_file = 'degree_target_train.csv'
    # target_test_file = 'degree_target_test.csv'
    # data_dir = '../datasets/census_processed/'
    # columns_list = COLUMNS

    source_train_file = 'undergrad_census9495_da_train.csv'
    target_train_file = 'grad_census9495_da_train.csv'
    target_test_file = 'grad_census9495_da_test.csv'

    # source_train_file = 'source_census9495_da_train.csv'
    # target_train_file = 'target_census9495_da_train.csv'
    # target_test_file = 'target_census9495_da_test.csv'

    data_dir = "/Users/yankang/Documents/Data/census/output/"

    columns_list = continuous_cols + categorical_cols + [target_col_name]
    print("columns_list:", len(columns_list), columns_list)

    file_dict = dict()
    file_dict['source_train_file'] = source_train_file
    file_dict['target_train_file'] = target_train_file
    file_dict['target_test_file'] = target_test_file
    file_dict['data_dir'] = data_dir
    train_on_dann(file_dict, columns=columns_list)
