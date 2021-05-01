import pandas as pd

from data_process.census_process.census_adult_process_utils import process_census_data, standardize_census_data

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_label"
]

ADULT_COLUMNS_NEW = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_label", "age_bucket"
]

RERANGED_ADULT_COLUMNS_NEW = [
    "age", "age_bucket", "workclass", "education", "education_num", "education_year",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "fnlwgt", "is_asian", "post_graduate", "income_label"
]


def process_census_adult_data(data_full_name, output_path, train=True):
    adult = pd.read_csv(data_full_name, names=ADULT_COLUMNS, skipinitialspace=True)
    print(f"adult shape:{adult.shape}")

    columns = adult.columns
    print(f"columns:{columns}")
    # print(adult[columns[0:8]].head())
    # print(adult[columns[8:]].head())
    print(adult[['age']])

    appendix = "_train" if train else "_test"
    appendix = appendix + ".csv"
    adult = process_census_data(adult)
    adult = adult[RERANGED_ADULT_COLUMNS_NEW]

    print(adult.shape)
    print(adult.head())
    adult.to_csv(output_path + 'processed_adult' + appendix, index=False)
    adult = standardize_census_data(adult)
    print(adult.head())
    adult.to_csv(output_path + 'standardized_adult' + appendix, index=False)


if __name__ == "__main__":
    data_path = "/Users/yankang/Documents/Data/census/"

    print("[INFO] process adult data")
    data_full_name = data_path + "adult.data"
    output_path = data_path + "output/"
    process_census_adult_data(data_full_name, output_path, train=True)

    print("[INFO] process adult test")
    data_full_name = data_path + "adult.test"
    process_census_adult_data(data_full_name, output_path, train=False)

    income_train_df = pd.read_csv(output_path + 'standardized_adult_train.csv', skipinitialspace=True)
    income_test_df = pd.read_csv(output_path + 'standardized_adult_test.csv', skipinitialspace=True)

    post_grad_train_df = income_train_df[income_train_df['post_graduate'] == 1]
    non_post_grad_train_df = income_train_df[income_train_df['post_graduate'] == 0]

    print("(S)ource:")
    print(f"train income_train_df shape:{income_train_df.shape}")
    print(f"train non_post_grad_train_df shape:{non_post_grad_train_df.shape}")
    print(f"train post_grad_train_df shape:{post_grad_train_df.shape}")

    post_grad_test_df = income_test_df[income_test_df['post_graduate'] == 1]
    non_post_grad_test_df = income_test_df[income_test_df['post_graduate'] == 0]

    print("(T)arget:")
    print(f"test income_test_df shape:{income_test_df.shape}")
    print(f"test non_post_grad_test_df shape:{non_post_grad_test_df.shape}")
    print(f"test post_grad_test_df shape:{post_grad_test_df.shape}")

    non_post_grad_train_df.to_csv('../datasets/census_processed/degree_source_train.csv', index=False)
    post_grad_train_df.to_csv('../datasets/census_processed/degree_target_train.csv', index=False)

    non_post_grad_test_df.to_csv('../datasets/census_processed/degree_source_test.csv', index=False)
    post_grad_test_df.to_csv('../datasets/census_processed/degree_target_test.csv', index=False)

    # adult_train = adult_train_df.values
    # adult_train_2 = adult_test_df[:5952].values
    # adult_test = adult_test_df[5952:].values
    # adult_train = np.concatenate([adult_train, adult_train_2], axis=0)
    #
    # adult_train = shuffle(adult_train)
    # adult_test = shuffle(adult_test)
    #
    # new_adult_train_df = pd.DataFrame(columns=adult_train_df.columns, data=adult_train)
    # new_adult_test_df = pd.DataFrame(columns=adult_train_df.columns, data=adult_test)
    #
    # asia_adult_train = new_adult_train_df[new_adult_train_df['is_asian'] == 1]
    # non_asia_adult_train = new_adult_train_df[new_adult_train_df['is_asian'] == 0]
    # print(f"asian train shape:{asia_adult_train.shape}")
    # print(f"non-asian train shape:{non_asia_adult_train.shape}")
    #
    # asia_adult_test = new_adult_test_df[new_adult_test_df['is_asian'] == 1]
    # non_asia_adult_test = new_adult_test_df[new_adult_test_df['is_asian'] == 0]
    # print(f"asian test shape:{asia_adult_test.shape}")
    # print(f"non-asian test shape:{non_asia_adult_test.shape}")
    #
    # non_asia_adult_train.to_csv('../datasets/census_processed/adult_source_train.csv', index=False)
    # asia_adult_train.to_csv('../datasets/census_processed/adult_target_train.csv', index=False)
    #
    # non_asia_adult_test.to_csv('../datasets/census_processed/adult_source_test.csv', index=False)
    # asia_adult_test.to_csv('../datasets/census_processed/adult_target_test.csv', index=False)
