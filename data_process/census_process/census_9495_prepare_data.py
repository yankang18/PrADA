import pandas as pd
from data_process.census_process.census_9495_process_utils import consistentize_census9495_columns, numericalize_census9495_data, \
    standardize_census9495_data, assign_native_country_identifier
from data_process.census_process.mapping_resource import cate_to_index_map, continuous_cols, categorical_cols, target_col_name
import numpy as np

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
    print("--> load original data")
    census = pd.read_csv(data_path, names=CENSUS_COLUMNS, skipinitialspace=True)
    print(census.head())
    print(census.shape)

    appendix = "_train" if train else "_test"
    extension = ".csv"
    appendix = appendix + extension
    print("--> consistentize original data")
    c_census = consistentize_census9495_columns(census)
    print(c_census.head())
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

    standardize_census9495_data(target_census_df, continuous_cols)
    standardize_census9495_data(source_census_df, continuous_cols)

    target_file_full_path = from_dir + 'target_census9495_da' + appendix
    source_file_full_path = from_dir + 'source_census9495_da' + appendix
    target_census_df.to_csv(target_file_full_path, header=True, index=False)
    source_census_df.to_csv(source_file_full_path, header=True, index=False)

    print(f"[INFO] saved target data to {target_file_full_path}")
    print(f"[INFO] saved source data to {source_file_full_path}")


def create_degree_source_target_data(p_census, from_dir, train=True, selected=True):
    appendix = create_file_appendix(train)

    doctorate_census = p_census[p_census['education'] == 11]
    master_census = p_census[(p_census['education'] == 9) | (p_census['education'] == 10)]
    undergrad_census = p_census[
        (p_census['education'] != 9) & (p_census['education'] != 10) & (p_census['education'] != 11)]
    columns = continuous_cols + categorical_cols + ['instance_weight', target_col_name]
    doctorate_census = doctorate_census[columns]
    master_census = master_census[columns]
    undergrad_census = undergrad_census[columns]
    print("doctorate_census shape", doctorate_census.shape)
    print("master_census shape", master_census.shape)
    print("undergrad_census shape", undergrad_census.shape)

    doctorate_census.to_csv(to_dir + 'doctorate_census9495' + appendix, header=True, index=False)
    master_census.to_csv(to_dir + 'master_census9495' + appendix, header=True, index=False)
    undergrad_census.to_csv(to_dir + 'undergrad_census9495' + appendix, header=True, index=False)

    doctorate_census = pd.read_csv(from_dir + 'doctorate_census9495' + appendix, skipinitialspace=True)
    master_census = pd.read_csv(from_dir + 'master_census9495' + appendix, skipinitialspace=True)
    undergrad_census = pd.read_csv(from_dir + 'undergrad_census9495' + appendix, skipinitialspace=True)
    compute_instance_prob(doctorate_census)
    compute_instance_prob(master_census)
    compute_instance_prob(undergrad_census)

    if selected:
        num_doctorate = doctorate_census.shape[0]
        print("[INFO] num_doctorate:", num_doctorate)
        doctorate_ins_prob = list(doctorate_census['instance_weight'].values)
        doctorate_index_arr = np.random.choice(a=np.arange(num_doctorate), size=2000, p=doctorate_ins_prob)

        num_master = master_census.shape[0]
        print("[INFO] num_master:", num_master)
        master_ins_prob = list(master_census['instance_weight'].values)
        master_index_arr = np.random.choice(a=np.arange(num_master), size=1400, p=master_ins_prob)

        num_undergrad = undergrad_census.shape[0]
        print("[INFO] num_undergrad:", num_undergrad)
        undergrad_ins_prob = list(undergrad_census['instance_weight'].values)
        undergrad_index_arr = np.random.choice(a=np.arange(num_undergrad), size=60000, p=undergrad_ins_prob)

        print("[INFO] doctorate_index_arr:", doctorate_index_arr.shape, doctorate_index_arr)
        print("[INFO] master_index_arr:", master_index_arr.shape, master_index_arr)
        print("[INFO] undergrad_index_arr:", undergrad_index_arr.shape, undergrad_index_arr)

        columns = continuous_cols + categorical_cols + [target_col_name]
        print("[INFO] number of columns:", len(columns), columns)

        doctorate_census_values = doctorate_census[columns].values[doctorate_index_arr]
        master_census_values = master_census[columns].values[master_index_arr]
        undergrad_census_values = undergrad_census[columns].values[undergrad_index_arr]
    else:
        doctorate_census_values = doctorate_census[columns].values
        master_census_values = master_census[columns].values
        undergrad_census_values = undergrad_census[columns].values

    # grad_census_values = np.concatenate([doctorate_census_values, master_census_values], axis=0)
    grad_census_values = doctorate_census_values
    grad_census_df = pd.DataFrame(data=grad_census_values, columns=columns)
    undergrad_census_df = pd.DataFrame(data=undergrad_census_values, columns=columns)

    print("grad_census_df shape:", grad_census_df.shape, grad_census_df[grad_census_df[target_col_name] == 1].shape)
    print("undergrad_census_df shape:", undergrad_census_df.shape,
          undergrad_census_df[undergrad_census_df[target_col_name] == 1].shape)

    standardize_census9495_data(grad_census_df, continuous_cols)
    standardize_census9495_data(undergrad_census_df, continuous_cols)

    grad_file_full_path = from_dir + 'grad_census9495_da' + appendix
    undergrad_file_full_path = from_dir + 'undergrad_census9495_da' + appendix
    grad_census_df.to_csv(grad_file_full_path, header=True, index=False)
    undergrad_census_df.to_csv(undergrad_file_full_path, header=True, index=False)

    print(f"[INFO] saved grad data to {grad_file_full_path}")
    print(f"[INFO] saved undergrad data to {undergrad_file_full_path}")


if __name__ == "__main__":
    train_data_path = "../../datasets/census_original/census-income.data"
    test_data_path = "../../datasets/census_original/census-income.test"
    to_dir = "../../datasets/census_processed/"

    train_df = process(train_data_path, to_dir=to_dir, train=True)
    test_df = process(test_data_path, to_dir=to_dir, train=False)

    create_degree_data = True
    if create_degree_data:
        create_degree_source_target_data(train_df, from_dir=to_dir, train=True, selected=True)
        create_degree_source_target_data(test_df, from_dir=to_dir, train=False, selected=False)
    else:
        create_asia_source_target_data(train_df, from_dir=to_dir, train=True, selected=True)
        create_asia_source_target_data(test_df, from_dir=to_dir, train=False, selected=False)

