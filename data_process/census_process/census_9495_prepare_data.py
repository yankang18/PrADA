import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from data_process.census_process.census_9495_process_utils import consistentize_census9495_columns, \
    numericalize_census9495_data, \
    standardize_census_data, assign_native_country_identifier
from data_process.census_process.mapping_resource import cate_to_index_map, continuous_cols, categorical_cols, \
    target_col_name

# CENSUS_COLUMNS = ["age", "class_worker", "det_ind_code", "det_occ_code", "education",
#                   "wage_per_hour", "hs_college", "marital_status", "major_ind_code", "occupation",
#                   "race", "hisp_origin", "gender", "union_member", "unemp_reason", "full_or_part_emp",
#                   "capital_gain", "capital_loss", "stock_dividends", "tax_filer_stat",
#                   "region_prev_res", "state_prev_res", "det_hh_fam_stat", "det_hh_summ", "instance_weight",
#                   "mig_chg_msa", "mig_chg_reg", "mig_move_reg", "mig_same", "mig_prev_sunbelt",
#                   "num_emp", "parent_status", "country_father", "country_mother", "native_country",
#                   "citizenship", "own_or_self", "vet_question", "vet_benefits", "weeks_worked",
#                   "year", "income_label"]

CENSUS_COLUMNS = ["age", "class_worker", "det_ind_code", "det_occ_code", "education",
                  "wage_per_hour", "hs_college", "marital_stat", "major_ind_code", "major_occ_code",
                  "race", "hisp_origin", "gender", "union_member", "unemp_reason", "full_or_part_emp",
                  "capital_gain", "capital_loss", "stock_dividends", "tax_filer_stat",
                  "region_prev_res", "state_prev_res", "det_hh_fam_stat", "det_hh_summ", "instance_weight",
                  "mig_chg_msa", "mig_chg_reg", "mig_move_reg", "mig_same", "mig_prev_sunbelt",
                  "num_emp", "fam_under_18", "country_father", "country_mother", "country_self",
                  "citizenship", "own_or_self", "vet_question", "vet_benefits", "weeks_worked",
                  "year", "income_label"]

# CENSUS_COLUMNS_NEW = ["age", "workclass", "det_ind_code", "det_occ_code", "education",
#                       "wage_per_hour", "hs_college", "marital_status", "major_ind_code", "major_occ_code",
#                       "race", "hisp_origin", "gender", "union_member", "unemp_reason",
#                       "full_or_part_emp", "capital_gain", "capital_loss", "stock_dividends", "tax_filer_stat",
#                       "region_prev_res", "state_prev_res", "det_hh_fam_stat", "det_hh_summ", "instance_weight",
#                       "mig_chg_msa", "mig_chg_reg", "mig_move_reg", "mig_same", "mig_prev_sunbelt",
#                       "num_emp", "parent_status", "country_father", "country_mother", "country_self",
#                       "citizenship", "own_or_self", "vet_question", "vet_benefits", "weeks_worked",
#                       "year", "income_label", "education_num", "age_bucket"]

RERANGED_CENSUS_COLUMNS_NEW = ["age", "gender_index", "age_index", "class_worker", "det_ind_code", "det_occ_code",
                               "education",
                               "education_year", "wage_per_hour", "hs_college", "marital_stat", "major_ind_code",
                               "major_occ_code", "race", "hisp_origin", "gender", "union_member", "unemp_reason",
                               "full_or_part_emp", "capital_gain", "capital_loss", "stock_dividends", "tax_filer_stat",
                               "region_prev_res", "state_prev_res", "det_hh_fam_stat", "det_hh_summ", "instance_weight",
                               "mig_chg_msa", "mig_chg_reg", "mig_move_reg", "mig_same", "mig_prev_sunbelt",
                               "num_emp", "fam_under_18", "country_father", "country_mother", "country_self",
                               "citizenship", "own_or_self", "vet_question", "vet_benefits", "weeks_worked",
                               "year", "income_label"]


def process(data_path, to_dir=None, train=True):
    census = pd.read_csv(data_path, names=CENSUS_COLUMNS, skipinitialspace=True)
    print("[INFO] load {} data".format("train" if train else "test"))
    print("[INFO] load data with shape:", census.shape)

    appendix = "_train" if train else "_test"
    extension = ".csv"
    appendix = appendix + extension

    print("--> consistentize original data")
    c_census = consistentize_census9495_columns(census)
    # print(c_census.head())
    c_census.to_csv(to_dir + 'consistentized_census9495' + appendix, header=True, index=False)

    print("--> numericalize data")
    p_census = numericalize_census9495_data(c_census, cate_to_index_map)
    return p_census


def compute_instance_prob(data_frame):
    weight_sum = data_frame["instance_weight"].sum()
    data_frame["instance_weight"] = data_frame["instance_weight"] / weight_sum


def create_file_appendix(train):
    appendix = "_train" if train else "_test"
    extension = ".csv"
    return appendix + extension


def create_asia_source_target_data(census_df, from_dir, train=True, selected=True):
    appendix = create_file_appendix(train)

    census_df['is_asian'] = census_df.apply(lambda row: assign_native_country_identifier(row.country_self), axis=1)

    target_census = census_df[census_df['is_asian'] == 1]
    source_census = census_df[census_df['is_asian'] == 0]
    print("target_census shape", target_census.shape)
    print("source_census shape", source_census.shape)
    compute_instance_prob(target_census)
    compute_instance_prob(source_census)

    if selected:
        num_target = target_census.shape[0]
        print("[INFO] num_target:", num_target)
        target_ins_prob = list(target_census['instance_weight'].values)
        target_index_arr = np.random.choice(a=np.arange(num_target), size=2000, p=target_ins_prob)

        num_source = source_census.shape[0]
        print("[INFO] num_source:", num_source)
        source_ins_prob = list(source_census['instance_weight'].values)
        source_index_arr = np.random.choice(a=np.arange(num_source), size=60000, p=source_ins_prob)

        # print("[INFO] doctorate_index_arr:", doctorate_index_arr.shape, doctorate_index_arr)
        print("[INFO] target_index_arr:", target_index_arr.shape, target_index_arr)
        print("[INFO] source_index_arr:", source_index_arr.shape, source_index_arr)

        columns = continuous_cols + categorical_cols + [target_col_name]
        print("[INFO] number of columns:", len(columns), columns)

        target_census_values = target_census[columns].values[target_index_arr]
        source_census_values = source_census[columns].values[source_index_arr]
    else:
        columns = continuous_cols + categorical_cols + [target_col_name]
        print("[INFO] number of columns:", len(columns), columns)

        target_census_values = target_census[columns].values
        source_census_values = source_census[columns].values

    target_census_df = pd.DataFrame(data=target_census_values, columns=columns)
    source_census_df = pd.DataFrame(data=source_census_values, columns=columns)

    print("target_census_df shape:", target_census_df.shape,
          target_census_df[target_census_df[target_col_name] == 1].shape)
    print("source_census_df shape:", source_census_df.shape,
          source_census_df[source_census_df[target_col_name] == 1].shape)

    standardize_census_data(target_census_df, continuous_cols)
    standardize_census_data(source_census_df, continuous_cols)

    target_file_full_path = from_dir + 'target_census9495_da' + appendix
    source_file_full_path = from_dir + 'source_census9495_da' + appendix
    target_census_df.to_csv(target_file_full_path, header=True, index=False)
    source_census_df.to_csv(source_file_full_path, header=True, index=False)

    print(f"[INFO] saved target data to {target_file_full_path}")
    print(f"[INFO] saved source data to {source_file_full_path}")


def create_degree_source_target_data(p_census, from_dir, to_dir, train=True, selected=None,
                                     grad_train_scaler=None, undergrad_train_scaler=None):
    appendix = create_file_appendix(train)
    print("--- create_degree_source_target_data for {} data --- ".format("train" if train else "test"))

    doctorate_census = p_census[p_census['education'] == 11]
    # master_census = p_census[(p_census['education'] == 9) | (p_census['education'] == 10)]
    master_census = p_census[p_census['education'] == 9]
    undergrad_census = p_census[
        (p_census['education'] != 9) & (p_census['education'] != 10) & (p_census['education'] != 11)]
    columns = continuous_cols + categorical_cols + ['instance_weight', target_col_name]
    doctorate_census = doctorate_census[columns]
    master_census = master_census[columns]
    undergrad_census = undergrad_census[columns]
    print("[INFO] doctorate_census shape", doctorate_census.shape)
    print("[INFO] master_census shape", master_census.shape)
    print("[INFO] undergrad_census shape", undergrad_census.shape)

    doctorate_census.to_csv(to_dir + 'doctorate_census9495' + appendix, header=True, index=False)
    master_census.to_csv(to_dir + 'master_census9495' + appendix, header=True, index=False)
    undergrad_census.to_csv(to_dir + 'undergrad_census9495' + appendix, header=True, index=False)

    doctorate_census = pd.read_csv(from_dir + 'doctorate_census9495' + appendix, skipinitialspace=True)
    master_census = pd.read_csv(from_dir + 'master_census9495' + appendix, skipinitialspace=True)
    undergrad_census = pd.read_csv(from_dir + 'undergrad_census9495' + appendix, skipinitialspace=True)

    if selected:
        compute_instance_prob(doctorate_census)
        compute_instance_prob(master_census)
        compute_instance_prob(undergrad_census)

        doctor_size = selected["doctor_size"]
        master_size = selected["master_size"]
        undergrad_size = selected["undergrad_size"]

        num_doctorate = doctorate_census.shape[0]
        print("[INFO] num_doctorate:", num_doctorate)
        doctorate_ins_prob = list(doctorate_census['instance_weight'].values)
        doctorate_index_arr = np.random.choice(a=np.arange(num_doctorate), size=doctor_size, p=doctorate_ins_prob)

        num_master = master_census.shape[0]
        print("[INFO] num_master:", num_master)
        master_ins_prob = list(master_census['instance_weight'].values)
        master_index_arr = np.random.choice(a=np.arange(num_master), size=master_size, p=master_ins_prob)

        num_undergrad = undergrad_census.shape[0]
        print("[INFO] num_undergrad:", num_undergrad)
        undergrad_ins_prob = list(undergrad_census['instance_weight'].values)
        undergrad_index_arr = np.random.choice(a=np.arange(num_undergrad), size=undergrad_size, p=undergrad_ins_prob)

        print("[INFO] doctorate_index_arr:", doctorate_index_arr.shape)
        print("[INFO] master_index_arr:", master_index_arr.shape)
        print("[INFO] undergrad_index_arr:", undergrad_index_arr.shape)

        columns = continuous_cols + categorical_cols + [target_col_name]
        print("[INFO] number of columns:", len(columns), columns)

        # doctorate_census_values = doctorate_census[columns].values[doctorate_index_arr]
        doctorate_census_values = doctorate_census[columns].values
        master_census_values = master_census[columns].values[master_index_arr]
        undergrad_census_values = undergrad_census[columns].values[undergrad_index_arr]
    else:
        doctorate_census_values = doctorate_census[columns].values
        master_census_values = master_census[columns].values
        undergrad_census_values = undergrad_census[columns].values

    grad_census_values = np.concatenate([doctorate_census_values, master_census_values], axis=0)
    # grad_census_values = doctorate_census_values

    # grad_census_values = shuffle(grad_census_values)
    # undergrad_census_values = shuffle(undergrad_census_values)

    grad_census_df = pd.DataFrame(data=grad_census_values, columns=columns)
    undergrad_census_df = pd.DataFrame(data=undergrad_census_values, columns=columns)

    grad_census_df_1 = grad_census_df[grad_census_df[target_col_name] == 1]
    grad_census_df_0 = grad_census_df[grad_census_df[target_col_name] == 0]

    undergrad_census_df_1 = undergrad_census_df[undergrad_census_df[target_col_name] == 1]
    undergrad_census_df_0 = undergrad_census_df[undergrad_census_df[target_col_name] == 0]

    print("(orig) grad_census_df_1 shape:", grad_census_df_1.shape)
    print("(orig) grad_census_df_0 shape:", grad_census_df_0.shape)
    print("(orig) undergrad_census_df_1 shape:", undergrad_census_df_1.shape)
    print("(orig) undergrad_census_df_0 shape:", undergrad_census_df_0.shape)

    num_pos = 200
    num_neg = 4000 - num_pos
    if train:
        grad_census_values_1 = grad_census_df_1.values[:num_pos]
        grad_census_values_0 = grad_census_df_0.values[:num_neg]
    else:
        grad_census_values_1 = grad_census_df_1.values
        grad_census_values_0 = grad_census_df_0.values

    print("grad_census_values_1 shape:", grad_census_values_1.shape)
    print("grad_census_values_0 shape:", grad_census_values_0.shape)
    print("undergrad_census_df_1 shape:", undergrad_census_df_1.shape)
    print("undergrad_census_df_0 shape:", undergrad_census_df_0.shape)

    grad_census_values_all = shuffle(np.concatenate((grad_census_values_1, grad_census_values_0), axis=0))
    grad_census_df_all = pd.DataFrame(data=grad_census_values_all, columns=columns)

    undergrad_census_values_all = shuffle(np.concatenate((undergrad_census_df_1, undergrad_census_df_0), axis=0))
    if train:
        undergrad_census_values_all = undergrad_census_values_all[:80000]

    undergrad_census_df_all = pd.DataFrame(data=undergrad_census_values_all, columns=columns)

    _, grad_train_scaler = standardize_census_data(grad_census_df_all, continuous_cols, grad_train_scaler)
    _, udgrad_train_scaler = standardize_census_data(undergrad_census_df_all, continuous_cols, undergrad_train_scaler)

    print("[INFO] (final) grad_census_df_all shape:", grad_census_df_all.shape)
    print("[INFO] (final) undergrad_census_df_all shape:", undergrad_census_df_all.shape)

    # save data
    grad_file_full_path = from_dir + 'grad_census9495_da_' + str(num_pos) + appendix
    undergrad_file_full_path = from_dir + 'undergrad_census9495_da_' + str(num_pos) + appendix
    grad_census_df_all.to_csv(grad_file_full_path, header=True, index=False)
    undergrad_census_df_all.to_csv(undergrad_file_full_path, header=True, index=False)
    print(f"[INFO] saved grad data to {grad_file_full_path}")
    print(f"[INFO] saved undergrad data to {undergrad_file_full_path}")

    return grad_train_scaler, udgrad_train_scaler


if __name__ == "__main__":
    data_path = "/Users/yankang/Documents/Data/census/"
    output_path = data_path + "output/"

    print("[INFO] ------ process data ------")
    train_data_path = data_path + "census-income.data"
    test_data_path = data_path + "census-income.test"
    train_df = process(train_data_path, to_dir=output_path, train=True)
    test_df = process(test_data_path, to_dir=output_path, train=False)

    # train_selected = {"doctor_size": 1200, "master_size": 1800, "undergrad_size": 60000}
    # test_selected = {"doctor_size": 600, "master_size": 1200, "undergrad_size": 60000}
    train_selected = None
    test_selected = None

    create_degree_data = True
    if create_degree_data:
        print("[INFO] ------ create Degree data ------")
        print("[INFO] ------ create Degree train data ------")
        grad_train_scaler, udgrad_train_scaler = create_degree_source_target_data(train_df,
                                                                                  from_dir=output_path,
                                                                                  to_dir=output_path,
                                                                                  train=True,
                                                                                  selected=train_selected)
        print("[INFO] ------ create Degree test data ------")
        create_degree_source_target_data(test_df,
                                         from_dir=output_path,
                                         to_dir=output_path,
                                         train=False,
                                         selected=test_selected,
                                         grad_train_scaler=grad_train_scaler,
                                         undergrad_train_scaler=udgrad_train_scaler)
    else:
        print("[INFO] ------ create Asia data ------")
        print("[INFO] ------ create Asia train data ------")
        create_asia_source_target_data(train_df, from_dir=output_path, train=True, selected=True)
        print("[INFO] ------ create Asia test data ------")
        create_asia_source_target_data(test_df, from_dir=output_path, train=False, selected=False)
