import numpy as np
import pandas as pd
from sklearn.utils import shuffle

feature_names = ['age',
                 'education_year',
                 'capital_gain',
                 'capital_loss',
                 'age_bucket',
                 'marital_status',
                 'gender',
                 'native_country',
                 'race',
                 'workclass',
                 'occupation',
                 'education']

COLUMNS_TO_LOAD = ['age',                 # 0
                   'education_year',      # 1
                   'capital_gain',        # 2
                   'capital_loss',        # 3
                   'age_bucket',          # 4
                   'marital_status',      # 5
                   'gender',              # 6
                   'native_country',      # 7
                   'race',                # 8
                   'workclass',           # 9
                   'occupation',          # 10
                   'education',           # 11
                   # "hours_per_week",      # 12
                   "relationship",        # 13
                   "income_label"]

label_name = "income_label"


def create_federated_learning_train_data(data_dict):
    print("[INFO] create_federated_learning_train_data")

    source_train = data_dict["source_train"]
    target_train = data_dict["target_train"]
    sampled_target_train = data_dict["sampled_target_train"]

    source_feat_train = source_train[feature_names].values
    source_label_train = source_train[label_name].values.reshape(-1, 1)
    target_feat_train = target_train[feature_names].values
    target_label_train = target_train[label_name].values.reshape(-1, 1)
    sampled_target_feat_train = sampled_target_train[feature_names].values
    sampled_target_label_train = sampled_target_train[label_name].values.reshape(-1, 1)

    print("source_feat_train:", source_feat_train.shape)
    print("source_label_train:", source_label_train.shape)
    print("target_feat_train:", target_feat_train.shape)
    print("target_label_train:", target_label_train.shape)
    print("sampled_target_feat_train:", sampled_target_feat_train.shape)
    print("sampled_target_label_train:", sampled_target_label_train.shape)

    comb_feat_train = np.concatenate([source_feat_train, sampled_target_feat_train], axis=1)
    print(f"concatenated data: {comb_feat_train.shape}")

    s_columns = ["s_" + col for col in feature_names]
    t_columns = ["t_" + col for col in feature_names]
    comb_columns = s_columns + t_columns
    print(f"s_columns:{s_columns}")
    print(f"t_columns:{t_columns}")
    print(f"comb_columns:{comb_columns}")
    host_dann_train_df = pd.DataFrame(data=comb_feat_train, columns=comb_columns)
    guest_dann_train_df = pd.DataFrame(data=source_label_train, columns=["y"])

    host_ft_train_df = pd.DataFrame(data=target_feat_train, columns=t_columns)
    guest_ft_train_df = pd.DataFrame(data=target_label_train, columns=["y"])

    return host_dann_train_df, guest_dann_train_df, host_ft_train_df, guest_ft_train_df


def create_federated_learning_test_data(data_dict):
    print("[INFO] create_federated_learning_test_data")

    target_test = data_dict["target_test"]
    target_test = shuffle(target_test)

    print(f"source_train.columns:{list(target_test.columns)}")
    feature_names = list(target_test.columns)
    feature_names.remove(label_name)
    print(f"feature names:{feature_names}")

    target_feat_test = target_test[feature_names].values
    target_label_test = target_test[label_name].values.reshape(-1, 1)

    print("target_feat_test:", target_feat_test.shape)
    print("target_label_test:", target_label_test.shape)

    t_columns = ["t_" + col for col in feature_names]

    host_test_df = pd.DataFrame(data=target_feat_test, columns=t_columns)
    guest_test_df = pd.DataFrame(data=target_label_test, columns=["y"])

    return host_test_df, guest_test_df


def create_host_only_train_data(source_feat_train, source_label_train, target_feat_train, target_label_train):
    print("source_feat_train:", source_feat_train.shape)
    print("source_label_train:", source_label_train.shape)
    print("target_feat_train:", target_feat_train.shape)
    print("target_label_train:", target_label_train.shape)

    host_train_data = np.concatenate([source_label_train, source_feat_train, target_feat_train], axis=1)
    print(f"host_train_data:{host_train_data.shape}")

    label_name = ["y"]
    s_columns = ["s_" + col for col in feature_names]
    t_columns = ["t_" + col for col in feature_names]
    host_dann_train_columns = label_name + s_columns + t_columns
    print(f"s_columns:{s_columns}")
    print(f"t_columns:{t_columns}")
    print(f"host dann train_columns:{host_dann_train_columns}")
    host_dann_comb_train_df = pd.DataFrame(data=host_train_data, columns=host_dann_train_columns)
    host_dann_comb_train_df = host_dann_comb_train_df.astype({'y': np.int32})

    host_target_train_df = pd.DataFrame(data=target_feat_train, columns=t_columns)
    guest_target_train_df = pd.DataFrame(data=target_label_train, columns=label_name)

    print(f"host_dann_comb_train_df shape:{host_dann_comb_train_df.shape}")
    print(f"host_target_train_df shape:{host_target_train_df.shape}")
    print(f"guest_target_train_df shape:{guest_target_train_df.shape}")

    return host_dann_comb_train_df, host_target_train_df, guest_target_train_df


def create_income_census_train_data(create_parties_train_data):
    # census_adult_file_name = './datasets/census_processed/standardized_adult.csv'
    # census_95_file_name = './datasets/census_processed/sampled_standardized_census95.csv'

    census_adult_file_name = '../../datasets/census_processed/standardized_adult.csv'
    census_95_file_name = '../../datasets/census_processed/sampled_standardized_census95.csv'

    source_train = pd.read_csv(census_adult_file_name, skipinitialspace=True)
    target_train = pd.read_csv(census_95_file_name, skipinitialspace=True)
    source_feat_train = source_train[feature_names].values
    source_label_train = source_train[label_name].values.reshape(-1, 1)
    target_feat_train = target_train[feature_names].values
    target_label_train = target_train[label_name].values.reshape(-1, 1)

    return create_parties_train_data(source_feat_train, source_label_train, target_feat_train, target_label_train)


def create_host_only_test_data(target_feat_test, target_label_test):
    label_name = ["y"]
    t_columns = ["t_" + col for col in feature_names]
    all_columns = label_name + t_columns

    print(f"host test comb_columns:{all_columns}")
    host_test_data = np.concatenate([target_label_test, target_feat_test], axis=1)
    print(f"host_test_data:{host_test_data.shape}, {host_test_data}")

    host_dann_comb_test_df = pd.DataFrame(data=host_test_data, columns=all_columns)
    host_dann_comb_test_df = host_dann_comb_test_df.astype({'y': np.int32})

    host_target_test_df = pd.DataFrame(data=target_feat_test, columns=t_columns)
    guest_target_test_df = pd.DataFrame(data=target_label_test, columns=label_name)

    print(f"host_dann_comb_test_df shape:{host_dann_comb_test_df.shape}")
    print(f"host_target_test_df shape:{host_target_test_df.shape}")
    print(f"guest_target_test_df shape:{guest_target_test_df.shape}")

    return host_dann_comb_test_df, host_target_test_df, guest_target_test_df


def create_income_census_test_data(create_parties_test_data):
    # census_adult_test_file_name = './datasets/census_processed/standardized_adult_test.csv'
    # census_95_test_file_name = 'datasets/census_processed/sampled_standardized_census95_test.csv'

    # census_adult_test_file_name = '../datasets/census_processed/standardized_adult_test.csv'
    census_95_test_file_name = '../../datasets/census_processed/sampled_standardized_census95_test.csv'
    target_test = pd.read_csv(census_95_test_file_name, skipinitialspace=True)
    print("target test:", target_test.shape)
    # print(target_test.head(20))

    target_test = shuffle(target_test)
    target_feat_test = target_test[feature_names].values
    target_label_test = target_test[label_name].values.reshape(-1, 1)

    print("target_feat_test:", target_feat_test.shape)
    print("target_label_test:", target_label_test.shape)

    return create_parties_test_data(target_feat_test, target_label_test)


def create_host_dann_data():
    host_dann_comb_train_df, host_target_train_df, guest_target_train_df = create_income_census_train_data(
        create_parties_train_data=create_host_only_train_data)
    host_dann_comb_train_df.to_csv('../datasets/census_processed/host_only_census_train_data_1.csv', header=True,
                                   index=True,
                                   index_label="id")
    host_target_train_df.to_csv('../datasets/census_processed/host_target_census_train_data.csv', header=True,
                                index=True,
                                index_label="id")
    guest_target_train_df.to_csv('../datasets/census_processed/guest_target_census_train_data.csv', header=True,
                                 index=True,
                                 index_label="id")

    host_dann_comb_test_df, host_target_test_df, guest_target_test_df = create_income_census_test_data(
        create_parties_test_data=create_host_only_test_data)
    host_dann_comb_test_df.to_csv('../datasets/census_processed/host_only_census_test_data_1.csv', header=True,
                                  index=True,
                                  index_label="id")
    host_target_test_df.to_csv('../datasets/census_processed/host_target_census_test_data.csv', header=True,
                               index=True,
                               index_label="id")
    guest_target_test_df.to_csv('../datasets/census_processed/guest_target_census_test_data.csv', header=True,
                                index=True,
                                index_label="id")


def create_census_fed_dann_data():
    host_train_df, guest_train_df = create_income_census_train_data(
        create_parties_train_data=create_federated_learning_train_data)
    host_train_df.to_csv('../datasets/census_processed/fed_host_census_train_data.csv', header=True,
                         index=True,
                         index_label="id")
    guest_train_df.to_csv('../datasets/census_processed/fed_guest_census_train_data.csv', header=True,
                          index=True,
                          index_label="id")

    host_test_df, guest_test_df = create_income_census_test_data(
        create_parties_test_data=create_federated_learning_test_data)
    host_test_df.to_csv('../datasets/census_processed/fed_host_census_test_data.csv', header=True,
                        index=True,
                        index_label="id")
    guest_test_df.to_csv('../datasets/census_processed/fed_guest_census_test_data.csv', header=True,
                         index=True,
                         index_label="id")


def create_asia_census_train_data(create_parties_train_data):

    source_adult_train_file_name = '../../datasets/census_processed/adult_source_train.csv'
    target_adult_train_file_name = '../../datasets/census_processed/adult_target_train.csv'

    source_train = pd.read_csv(source_adult_train_file_name, skipinitialspace=True)
    target_train = pd.read_csv(target_adult_train_file_name, skipinitialspace=True)

    num_sample_target = target_train.shape[0]
    num_to_sample = source_train.shape[0] - num_sample_target

    sampled_target_train = None
    if num_to_sample > 0:
        num_to_sample_idx = np.random.choice(list(range(num_sample_target)), num_to_sample)
        print(f"num_to_sample_idx:{num_to_sample_idx}")
        sampled_target_values = target_train.values[num_to_sample_idx]
        sampled_target_values = np.concatenate([sampled_target_values, target_train.values], axis=0)
        sampled_target_values = shuffle(sampled_target_values)
        sampled_target_train = pd.DataFrame(data=sampled_target_values, columns=target_train.columns)

    print(f"source_train.columns:{list(source_train.columns)}")
    feature_names = list(source_train.columns)
    feature_names.remove(label_name)
    print(f"feature names:{feature_names}")

    data_dict = {"source_train": source_train,
                 "target_train": target_train,
                 "sampled_target_train": sampled_target_train}

    return create_parties_train_data(data_dict)


def create_asia_census_test_data(create_parties_test_data):

    target_adult_test_file_name = '../../datasets/census_processed/adult_target_test.csv'
    # source_adult_test_file_name = '../datasets/census_processed/adult_source_test.csv'

    target_test = pd.read_csv(target_adult_test_file_name, skipinitialspace=True)
    print("target test:", target_test.shape)
    # print(target_test.head(20))
    data_dict = {"target_test": target_test}

    return create_parties_test_data(data_dict)


def create_asia_fed_dann_data():
    host_dann_train_df, guest_dann_train_df, host_ft_train_df, guest_ft_train_df = create_asia_census_train_data(
        create_parties_train_data=create_federated_learning_train_data)
    host_dann_train_df.to_csv('../datasets/census_processed/fed_host_dann_asia_train_data.csv', header=True,
                              index=True,
                              index_label="id")
    guest_dann_train_df.to_csv('../datasets/census_processed/fed_guest_dann_asia_train_data.csv', header=True,
                               index=True,
                               index_label="id")

    host_ft_train_df.to_csv('../datasets/census_processed/fed_host_ft_asia_train_data.csv', header=True,
                            index=True,
                            index_label="id")
    guest_ft_train_df.to_csv('../datasets/census_processed/fed_guest_ft_asia_train_data.csv', header=True,
                             index=True,
                             index_label="id")

    host_test_df, guest_test_df = create_asia_census_test_data(
        create_parties_test_data=create_federated_learning_test_data)
    host_test_df.to_csv('../datasets/census_processed/fed_host_asia_test_data.csv', header=True,
                        index=True,
                        index_label="id")
    guest_test_df.to_csv('../datasets/census_processed/fed_guest_asia_test_data.csv', header=True,
                         index=True,
                         index_label="id")


if __name__ == "__main__":
    create_asia_fed_dann_data()
    # create_census_fed_dann_data()
