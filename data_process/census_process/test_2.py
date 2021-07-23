import pandas as pd

if __name__ == "__main__":
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    # source_adult_train_file_name = data_dir + 'undergrad_census9495_da_300_train.csv'
    # target_adult_train_file_name = data_dir + 'grad_census9495_da_300_train.csv'
    # source_adult_test_file_name = data_dir + 'undergrad_census9495_da_300_test.csv'
    # target_adult_test_file_name = data_dir + 'grad_census9495_da_300_test.csv'
    source_adult_train_file_name = data_dir + 'undergrad_census9495_da1623962998_train.csv'
    target_adult_pre_train_file_name = data_dir + 'grad_census9495_da1623962998_train.csv'
    target_adult_ft_train_file_name = data_dir + 'grad_census9495_ft1623962998_train.csv'
    source_adult_test_file_name = data_dir + 'undergrad_census9495_da1623963001_test.csv'
    target_adult_test_file_name = data_dir + 'grad_census9495_ft1623963001_test.csv'

    adult_source_train = pd.read_csv(source_adult_train_file_name, skipinitialspace=True)
    adult_pre_target_train = pd.read_csv(target_adult_pre_train_file_name, skipinitialspace=True)
    adult_ft_target_train = pd.read_csv(target_adult_ft_train_file_name, skipinitialspace=True)
    adult_source_test = pd.read_csv(source_adult_test_file_name, skipinitialspace=True)
    adult_target_test = pd.read_csv(target_adult_test_file_name, skipinitialspace=True)

    print('adult_source_train:', adult_source_train.shape)
    print('adult_pre_target_train:', adult_pre_target_train.shape)
    print('adult_ft_target_train:', adult_ft_target_train.shape)
    print('adult_source_test:', adult_source_test.shape)
    print('adult_target_test:', adult_target_test.shape)
