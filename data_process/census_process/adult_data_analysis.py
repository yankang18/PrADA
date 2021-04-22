import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#
# COLUMNS = ['age',
#            'education_year',
#            # 'hours_per_week',
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
#            "income_label"]
#
#
# def train_benchmark(samples_train, samples_test):
#     train_data, train_label = samples_train[:, :-1], samples_train[:, -1]
#     test_data, test_label = samples_test[:, :-1], samples_test[:, -1]
#
#     print(f"train_data shape : {train_data.shape}")
#     print(f"train_label shape : {train_label.shape}")
#     print(f"test_data shape : {test_data.shape}")
#     print(f"test_label shape : {test_label.shape}")
#     run_benchmark(train_data, train_label, test_data, test_label)
#
#     # print(f"train_data shape : {train_data.shape}")
#     # print(f"train_label shape : {train_label.shape}")
#     # print(f"test_data shape : {test_data.shape}")
#     # print(f"test_label shape : {test_label.shape}")
#     # # lr_adult = LogisticRegressionCV(cv=5, random_state=0, max_iter=500)
#     # lr_adult = LogisticRegression(max_iter=500)
#     # # lr_adult = LogisticRegression(max_iter=500)
#     # cls = lr_adult.fit(train_data, train_label)
#     # pred = cls.predict(test_data)
#     # test_label = test_label.astype(float)
#     # # print("pred:", pred, pred.shape, type(pred), sum(pred))
#     # # print("test_label:", test_label, test_label.shape, type(test_label), sum(test_label))
#     # # pred_prob = cls.predict_proba(test_data)
#     # # print("pred_prob shape:", pred_prob.shape)
#     # acc = accuracy_score(pred, test_label)
#     # auc = roc_auc_score(pred, test_label)
#     # res = precision_recall_fscore_support(pred, test_label, average='macro')
#     # print(f"accuracy : {acc}")
#     # print(f"auc : {auc}")
#     # print(f"prf : {res}")
#     # print(f"coef: {cls.coef_}")
#
#
# def train_census_adult():
#     adult = pd.read_csv('../datasets/census_processed/standardized_adult.csv', skipinitialspace=True)
#     adult = adult[COLUMNS]
#
#     adult_test = pd.read_csv('../datasets/census_processed/standardized_adult_test.csv', skipinitialspace=True)
#     adult_samples_test = adult_test[COLUMNS].values
#
#     # adult_samples = adult.values
#     # adult_samples_train, adult_samples_test = train_test_split(adult_samples, train_size=0.7)
#     adult_samples_train = adult.values
#
#     print("adult_samples_train", adult_samples_train.shape)
#     print("adult_samples_test", adult_samples_test.shape)
#     train_benchmark(adult_samples_train, adult_samples_test)
#
#
# def prepare_census_95_train_data(num_sample=1000):
#     census95 = pd.read_csv('../datasets/census_processed/sampled_standardized_census95.csv', skipinitialspace=True)
#     census95 = census95[COLUMNS]
#
#     census95_1 = census95[census95["income_label"] == 1]
#     census95_0 = census95[census95["income_label"] == 0]
#     census95_1 = shuffle(census95_1)
#     census95_0 = shuffle(census95_0)
#
#     print(f"original census95 with label 1 has shape:{census95_1.shape}")
#     print(f"original census95 with label 0 has shape:{census95_0.shape}")
#
#     num_sample_1 = int(num_sample / 4)
#     num_sample_0 = num_sample - num_sample_1
#     census95_1_500 = census95_1.values[:num_sample_1]
#     census95_0_500 = census95_0.values[:num_sample_0]
#     census95_1000 = np.concatenate([census95_1_500, census95_0_500], axis=0)
#     census95_1000 = shuffle(census95_1000)
#     print(f"census95_1_500 has shape:{census95_1_500.shape}")
#     print(f"census95_0_500 has shape:{census95_0_500.shape}")
#     print(f"census95_1000 has shape:{census95_1000.shape}")
#
#     census95_1000_df = pd.DataFrame(data=census95_1000, columns=COLUMNS)
#     census95_1000_df.to_csv('../datasets/census_processed/standardized_census95_' + "train_" + str(num_sample) + ".csv",
#                             index=False)
#     return census95_1000_df
#
#
# def train_census_95():
#     # census95 = pd.read_csv('../datasets/census_processed/sampled_standardized_census95.csv', skipinitialspace=True)
#     # census95 = pd.read_csv('../datasets/census_processed/standardized_census95_benchmark_train_9768.csv',
#     #                        skipinitialspace=True)
#     census95 = pd.read_csv('../datasets/census_processed/sampled_standardized_census95_train.csv',
#                            skipinitialspace=True)
#     census95 = census95[COLUMNS]
#     census95_samples_train = census95.values
#
#     census95_test = pd.read_csv('../datasets/census_processed/standardized_census95_test.csv',
#                                 skipinitialspace=True)
#     # census95_test = sample_data(census95_test, num_samples=15000)
#     census95_samples_test = census95_test[COLUMNS].values
#
#     print(census95[census95["income_label"] == 1].shape)
#     print(census95[census95["income_label"] == 0].shape)
#
#     # census95_samples_train, _ = train_test_split(census95_samples_train, train_size=0.3, shuffle=True)
#     # print("census95_samples_train", census95_samples_train.shape)
#     # print("census95_samples_test", census95_samples_test.shape)
#     # num_sample = census95_samples_train.shape[0]
#     # census95_1000_df = pd.DataFrame(data=census95_samples_train, columns=COLUMNS)
#     # print(census95_1000_df[census95_1000_df["income_label"] == 1].shape)
#     # print(census95_1000_df[census95_1000_df["income_label"] == 0].shape)
#     # census95_1000_df.to_csv(
#     #     '../datasets/census_processed/standardized_census95_benchmark_' + "train_" + str(num_sample) + ".csv",
#     #     index=False)
#
#     train_benchmark(census95_samples_train, census95_samples_test)
#     # train_benchmark(census95_1000, census95_samples_test)
#
#
# def train_adult_to_95():
#     adult = pd.read_csv('../datasets/census_processed/standardized_adult.csv', skipinitialspace=True)
#     adult = adult[COLUMNS]
#     census95 = pd.read_csv('../datasets/census_processed/sampled_standardized_census95.csv', skipinitialspace=True)
#     census95 = census95[COLUMNS]
#
#     adult_samples = adult.values
#     census95_samples = census95.values
#
#     census95_samples = shuffle(census95_samples)
#     ratio_of_95_to_train = 0.3
#     num_train = int(ratio_of_95_to_train * len(census95_samples))
#     census95_samples_train = census95_samples[:num_train]
#
#     census95_test = pd.read_csv('../datasets/census_processed/sampled_standardized_census95_test.csv',
#                                 skipinitialspace=True)
#     census95_samples_test = census95_test[COLUMNS].values
#     # census95_samples_test = census95_samples[-5000:]
#
#     print("census95_samples_train shape:", census95_samples_train.shape)
#
#     comb_samples_train = np.concatenate([adult_samples, census95_samples_train], axis=0)
#     comb_samples_train = shuffle(comb_samples_train)
#
#     train_benchmark(comb_samples_train, census95_samples_test)
#
#
# def train_adult():
#     adult_train = pd.read_csv('../datasets/census_processed/standardized_adult_train.csv', skipinitialspace=True)
#     adult_train = adult_train[COLUMNS]
#     adult_test = pd.read_csv('../datasets/census_processed/standardized_adult_test.csv', skipinitialspace=True)
#     adult_test = adult_test[COLUMNS]
#
#     adult_train = adult_train.values
#     adult_test = adult_test.values
#
#     adult_train = shuffle(adult_train)
#     adult_test = shuffle(adult_test)
#
#     train_benchmark(adult_train, adult_test)


if __name__ == "__main__":
    adult_train = pd.read_csv('../../datasets/census_processed/processed_adult_train.csv', skipinitialspace=True)

    print(f"columns:{adult_train.columns}")
    print(adult_train["education_year"].describe())
    print(f"asian shape:{adult_train[adult_train['is_asian']==1].shape}")
    print(f"non-asian shape:{adult_train[adult_train['is_asian'] == 0].shape}")

    asia_adult = adult_train[adult_train['is_asian'] == 1]

    print(f"asian income lbl 1 shape:{asia_adult[asia_adult['income_label'] == 1].shape}")
    print(f"asian income lbl 0 shape:{asia_adult[asia_adult['income_label'] == 0].shape}")

    print(f"asian shape:{adult_train[adult_train['income_label'] == 1].shape}")
    print(f"non-asian shape:{adult_train[adult_train['income_label'] == 0].shape}")

    bin_values = np.arange(start=0, stop=21, step=1)
    group_native_country = adult_train.groupby('is_asian')['occupation']
    print(group_native_country)
    # group_native_country.plot(kind='hist', bins=bin_values, figsize=[12, 6], alpha=.4, legend=True)  # alpha for transparency
    group_native_country.plot(kind='density', figsize=[12, 6], alpha=.4, legend=True)  # alpha for transparency
    plt.show()

