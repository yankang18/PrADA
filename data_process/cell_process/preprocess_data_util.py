import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_process.cell_process.create_data_tables_obsolete import get_tag_id2info_dicts


def normalize(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled


def partial_normalize(df, no_norm_cols, to_norm_cols):
    no_norm_values = df[no_norm_cols].values
    to_norm_values = df[to_norm_cols].values
    norm_values = normalize(to_norm_values)
    values = np.concatenate([no_norm_values, norm_values], axis=1)
    cols = no_norm_cols + to_norm_cols
    return pd.DataFrame(data=values, columns=cols)


def normalize_df(df):
    column_names = df.columns
    x = df.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    scaled_df = pd.DataFrame(data=x_scaled, columns=column_names)
    return scaled_df


class MissingFillTransformer(object):

    def __init__(self, filling_value=-99):
        super(MissingFillTransformer, self).__init__()
        self.filling_value = filling_value

    def transform(self, df_x):
        df_x = df_x.fillna(self.filling_value)
        assert df_x.isnull().sum().sum() == 0
        return df_x


class StandardTransformer(object):

    def __init__(self):
        super(StandardTransformer, self).__init__()

    def transform(self, df_x):
        return normalize_df(df_x)


class TransformPipeLine(object):
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def transform(self, df_x):
        for transformer in self.transformer_list:
            df_x = transformer.transform(df_x)
        return df_x


def preprocess_tables(from_dir, to_dir, app_table_names):
    for table_name in app_table_names:
        df = pd.read_csv(from_dir + table_name)

        df = df.fillna(-99)
        assert df.isnull().sum().sum() == 0

        norm_df = normalize_df(df)

        file_path = to_dir + "p_" + table_name
        norm_df.to_csv(file_path, index=False)
        print(f"save {table_name} to: {file_path}")


def preprocess(dir, to_dir):
    demo_file_name = "named_demographics.csv"
    named_demo_df = pd.read_csv(dir + demo_file_name)
    named_demo_df = named_demo_df.fillna(-99)
    assert named_demo_df.isnull().sum().sum() == 0
    norm_named_demo_df = normalize_df(named_demo_df)
    file_path = to_dir + "p_" + demo_file_name
    norm_named_demo_df.to_csv(file_path, index=False)
    print(f"save demographics to: {file_path}")

    equip_file_name = "named_equipment.csv"
    named_equip_df = pd.read_csv(dir + equip_file_name)
    named_equip_df = named_equip_df.fillna(-99)
    assert named_equip_df.isnull().sum().sum() == 0
    norm_named_equip_df = normalize_df(named_equip_df)
    file_path = to_dir + "p_" + equip_file_name
    norm_named_equip_df.to_csv(file_path, index=False)
    print(f"save equipment to: {file_path}")

    risk_file_name = "named_risk.csv"
    named_risk_df = pd.read_csv(dir + risk_file_name)
    named_risk_df = named_risk_df.fillna(-99)
    assert named_risk_df.isnull().sum().sum() == 0
    norm_named_risk_df = normalize_df(named_risk_df)
    file_path = to_dir + "p_" + risk_file_name
    norm_named_risk_df.to_csv(file_path, index=False)
    print(f"save risk to: {file_path}")

    appears_file_name = "named_cell_number_appears.csv"
    named_appears_df = pd.read_csv(dir + "named_cell_number_appears.csv")
    named_appears_df = named_appears_df.fillna(-99)
    assert named_appears_df.isnull().sum().sum() == 0
    norm_named_appears_df = normalize_df(named_appears_df)
    file_path = to_dir + "p_" + appears_file_name
    norm_named_appears_df.to_csv(file_path, index=False)
    print(f"save appears to: {file_path}")

    app_table_names = ["tgi.csv", "fraud.csv", "app_finance_install.csv", "app_finance_usage.csv",
                       "app_life_install.csv", "app_life_usage.csv"]
    preprocess_tables(dir, to_dir, app_table_names)


def preprocess_v2(dir, to_dir):
    demo_file_name = "named_demographics.csv"
    named_demo_df = pd.read_csv(dir + demo_file_name)
    # named_demo_df['年龄分段'] = named_demo_df['年龄分段'].fillna(0)
    # named_demo_df['原始年龄'] = named_demo_df['原始年龄'].fillna(0)
    # named_demo_df['学历'] = named_demo_df['学历'].fillna(7)
    # named_demo_df['年龄预测'] = named_demo_df['年龄预测'].fillna(6)
    # named_demo_df['资产属性'] = named_demo_df['资产属性'].fillna(5)
    # named_demo_df['收入水平'] = named_demo_df['收入水平'].fillna(5)
    # named_demo_df = named_demo_df.fillna(0)
    # named_demo_df = named_demo_df.fillna(-1)
    named_demo_df = named_demo_df.fillna(-99)
    assert named_demo_df.isnull().sum().sum() == 0
    # no_norm_cols = ["年龄分段", "年龄预测", "学历"]
    # to_norm_cols = ["原始年龄", "教育", "党员", "工薪族", "公务员"]
    # norm_named_demo_df = partial_normalize(named_demo_df, no_norm_cols, to_norm_cols)
    norm_named_demo_df = normalize_df(named_demo_df)
    file_path = to_dir + "p_" + demo_file_name
    norm_named_demo_df.to_csv(file_path, index=False)
    print(f"save demographics to: {file_path}")

    # asset_file_name = "named_asset.csv"
    # named_asset_df = pd.read_csv(dir + asset_file_name)
    # named_asset_df['资产属性'] = named_asset_df['资产属性'].fillna(5)
    # named_asset_df['收入水平'] = named_asset_df['收入水平'].fillna(5)
    # named_asset_df = named_asset_df.fillna(0)
    # assert named_asset_df.isnull().sum().sum() == 0
    # # no_norm_cols = ["资产属性", "收入水平"]
    # # to_norm_cols = ["收支流水分模型", "资产通用分", "信用卡额度分模型", "工资水平模型", "公积金分数模型", "信用卡额度扩散",
    # #                 "消费水平", "有车", "有房", "公积金"]
    # # norm_named_asset_df = partial_normalize(named_asset_df, no_norm_cols, to_norm_cols)
    # norm_named_asset_df = normalize_df(named_asset_df)
    # file_path = to_dir + "p_" + asset_file_name
    # norm_named_asset_df.to_csv(file_path, index=False)
    # print(f"save asset to: {file_path}")

    equip_file_name = "named_equipment.csv"
    named_equip_df = pd.read_csv(dir + equip_file_name)
    # named_equip_df['分辨率离散'] = named_equip_df['分辨率离散'].fillna(0)
    # named_equip_df['ram大小离散'] = named_equip_df['ram大小离散'].fillna(15)
    # named_equip_df['api_level'] = named_equip_df['api_level'].fillna(0)
    # named_equip_df['machine价格离散'] = named_equip_df['machine价格离散'].fillna(13)
    # named_equip_df['rom离散'] = named_equip_df['rom离散'].fillna(23)
    # named_equip_df['cpu频率离散'] = named_equip_df['cpu频率离散'].fillna(25)
    # named_equip_df['cpu核数离散'] = named_equip_df['cpu核数离散'].fillna(11)
    # named_equip_df = named_equip_df.fillna(0)
    named_equip_df = named_equip_df.fillna(-99)
    assert named_equip_df.isnull().sum().sum() == 0
    norm_named_equip_df = normalize_df(named_equip_df)
    file_path = to_dir + "p_" + equip_file_name
    norm_named_equip_df.to_csv(file_path, index=False)
    print(f"save equipment to: {file_path}")

    risk_file_name = "named_risk.csv"
    named_risk_df = pd.read_csv(dir + risk_file_name)
    # named_risk_df = named_risk_df.fillna(0)
    named_risk_df = named_risk_df.fillna(-99)
    assert named_risk_df.isnull().sum().sum() == 0
    norm_named_risk_df = normalize_df(named_risk_df)
    file_path = to_dir + "p_" + risk_file_name
    norm_named_risk_df.to_csv(file_path, index=False)
    print(f"save risk to: {file_path}")

    appears_file_name = "named_cell_number_appears.csv"
    named_appears_df = pd.read_csv(dir + "named_cell_number_appears.csv")
    named_appears_df = named_appears_df.fillna(-99)
    assert named_appears_df.isnull().sum().sum() == 0
    norm_named_appears_df = normalize_df(named_appears_df)
    file_path = to_dir + "p_" + appears_file_name
    norm_named_appears_df.to_csv(file_path, index=False)
    print(f"save appears to: {file_path}")

    app_table_names = ["tgi.csv", "fraud.csv", "app_finance_install.csv", "app_finance_usage.csv",
                       "app_life_install.csv", "app_life_usage.csv"]
    preprocess_tables(dir, to_dir, app_table_names)


if __name__ == "__main__":
    id_mean_v4_df = pd.read_csv('../../../data/cell_manager/ID_meaning_v5.csv')
    id2desc_dict, id2type1_dict, id2type2_dict = get_tag_id2info_dicts(id_mean_v4_df)

    # a_dir = "../../../data/cell_manager/A_train_data/"
    # b_dir = "../../../data/cell_manager/B_train_data/"

    a_dir = "../../../data/cell_manager/A_train_data_2/"
    b_dir = "../../../data/cell_manager/B_train_data_2/"
    c_dir = "../../../data/cell_manager/C_train_data_2/"
    to_a_dir = "../../../data/cell_manager/A_train_data_2/"
    to_b_dir = "../../../data/cell_manager/B_train_data_2/"
    to_c_dir = "../../../data/cell_manager/C_train_data_2/"

    ## table_names = ['demographics.csv', 'asset.csv', 'equipment.csv', 'risk.csv', 'cell_number_appears.csv']
    # table_names = ['demographics.csv', 'equipment.csv', 'risk.csv', 'cell_number_appears.csv']
    # create_named_data_tables(a_dir, to_a_dir, table_names, id2desc_dict)
    # create_named_data_tables(b_dir, to_b_dir, table_names, id2desc_dict)
    # create_named_data_tables(c_dir, to_c_dir, table_names, id2desc_dict)
    # print("finished creating named tables!")

    # preprocess(a_dir, to_a_dir)
    # preprocess(b_dir, to_b_dir)
    preprocess(c_dir, to_c_dir)
    print("finished preprocessing data!")
