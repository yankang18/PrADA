from collections import OrderedDict

from datasets.ppd_dataloader import get_datasets, get_dataloader
from experiments.kesci_ppd.meta_data import column_name_list, group_ind_list, group_info, embedding_shape_map
from models.classifier import CensusRegionAggregator
from models.dann_models import create_embeddings
from models.discriminator import CensusRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from models.model_config import wire_global_model
from utils import get_timestamp, get_current_date


def record_domain_data(domain_data_dict, domain_data, domain_col_list, is_categorical):
    if is_categorical:
        emb_dict = OrderedDict()
        for col_index, col_name in enumerate(domain_col_list):
            emb_dict[col_name] = domain_data[:, col_index]
        domain_data_dict["embeddings"] = emb_dict

    else:
        domain_data_dict["non_embedding"] = {"tabular_data": domain_data}


def parse_domain_data(data, column_name_list, df_group_ind_list, df_group_info_list):
    wide_feat_list = []
    domain_list = []
    for group_ind, group_info in zip(df_group_ind_list, df_group_info_list):
        start_index = group_info[0]
        # 2 stands for wide features (i.e., features for active parties)
        if group_ind == 2:
            length, _ = group_info[1]
            wide_feat_list.append(data[:, start_index:start_index + length])
        # 1 stands for feature group (i.e., feature for passive party)
        elif group_ind == 1:
            new_domain = dict()
            new_domain["embeddings"] = None
            new_domain["non_embedding"] = None
            tuple_num = len(group_info)
            for i in range(1, tuple_num):
                length, is_cat = group_info[i]
                domain_data = data[:, start_index:start_index + length]
                domain_col_list = column_name_list[start_index:start_index + length]
                record_domain_data(new_domain, domain_data, domain_col_list, is_cat)
                start_index = start_index + length
            domain_list.append(new_domain)
    return wide_feat_list, domain_list


def partition_data(data):
    """
    partition data into party C and party A(B). Data in party C is partitioned into group
    """

    wide_feat_list, domain_list = parse_domain_data(data,
                                                    column_name_list,
                                                    group_ind_list,
                                                    group_info)
    return wide_feat_list, domain_list


def create_model_group(extractor_input_dim):
    """
    create a group of models, namely feature extractor, aggregator and discriminator, for each feature group
    """

    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dim)
    classifier = CensusRegionAggregator(input_dim=extractor_input_dim[-1])
    discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dim[-1])
    return extractor, classifier, discriminator


def create_embedding_dict():
    """
    create embedding dictionary for categorical features/columns
    """

    embedding_meta_dict = dict()
    for feat_name, emb_shape in embedding_shape_map.items():
        embedding_meta_dict[feat_name] = emb_shape
    print(f"embedding_meta_dict: \n {embedding_meta_dict}")
    return create_embeddings(embedding_meta_dict)


# def compute_feature_group_interaction(input_dims_list):
#     input_int_dims_list = [e for e in input_dims_list]
#     start_index = 1
#     for fg in input_dims_list:
#         for i in range(start_index, len(input_dims_list)):
#             fg_2 = input_dims_list[i]
#             input_int_dims_list.append([v1 + v2 for v1, v2 in zip(fg, fg_2)])
#         start_index += 1
#     return input_int_dims_list


def create_pdd_global_model(pos_class_weight=1.0):
    embedding_dict = create_embedding_dict()
    print("[INFO] embedding_dict", embedding_dict)

    input_dims_list = [[31, 40, 31, 8],
                       [15, 24, 15, 6],
                       [85, 110, 85, 10],
                       [30, 50, 30, 8],
                       [18, 30, 18, 6],
                       [55, 80, 55, 10]]
    # input_dims_list = [[15, 24, 15, 6],
    #                    [85, 100, 85, 10],
    #                    [27, 50, 27, 8],
    #                    [18, 30, 18, 6],
    #                    [55, 80, 55, 10]]

    num_wide_feature = 6
    # num_wide_feature = 17
    using_feature_group = True
    using_interaction = False
    using_transform_matrix = False

    global_model = wire_global_model(embedding_dict=embedding_dict,
                                     input_dims_list=input_dims_list,
                                     num_wide_feature=num_wide_feature,
                                     using_feature_group=using_feature_group,
                                     using_interaction=using_interaction,
                                     using_transform_matrix=using_transform_matrix,
                                     partition_data_fn=partition_data,
                                     create_model_group_fn=create_model_group,
                                     pos_class_weight=1.0)

    return global_model


if __name__ == "__main__":

    exp_dir = "ppd_dann"

    # data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    # source_train_file_name = data_dir + "PPD_2014_1to9_train.csv"
    # target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
    # source_test_file_name = data_dir + 'PPD_2014_1to9_test.csv'
    # target_test_file_name = data_dir + 'PPD_2014_10to12_test.csv'

    # data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output/"

    # source_train_file_name = data_dir+"PPD_2014_1to8_train.csv"
    # target_train_file_name = data_dir+'PPD_2014_10to12_train.csv'
    # source_test_file_name = data_dir+'PPD_2014_1to8_test.csv'
    # target_test_file_name = data_dir+'PPD_2014_10to12_test.csv'

    # source_train_file_name = data_dir + "PPD_2014_src_1to8_train.csv"
    # target_train_file_name = data_dir + 'PPD_2014_tgt_10to11_train.csv'
    # source_test_file_name = data_dir + 'PPD_2014_src_1to8_test.csv'
    # target_test_file_name = data_dir + 'PPD_2014_tgt_10to11_test.csv'

    # source_train_file_name = data_dir + "PPD_2014_src_1to9_train.csv"
    # source_test_file_name = data_dir + 'PPD_2014_src_1to9_test.csv'
    # source_train_file_name = data_dir + "PPD_2014_src_1to8_train.csv"
    # source_test_file_name = data_dir + 'PPD_2014_src_1to8_test.csv'
    # target_train_file_name = data_dir + 'PPD_2014_tgt_9_train.csv'
    # target_test_file_name = data_dir + 'PPD_2014_tgt_9_test.csv'
    # target_train_file_name = 'PPD_2014_tgt_10to11_train.csv'
    # target_test_file_name = 'PPD_2014_tgt_10to11_test.csv'

    timestamp = '1620085151'
    data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{timestamp}/"
    # source_train_file = "PPD_2014_src_1to9_train.csv"
    # source_test_file = 'PPD_2014_src_1to9_test.csv'
    source_train_file_name = data_dir + "PPD_2014_src_1to8_train.csv"
    source_test_file_name = data_dir + 'PPD_2014_src_1to8_test.csv'
    target_train_file_name = data_dir + 'PPD_2014_tgt_9_train.csv'
    target_test_file_name = data_dir + 'PPD_2014_tgt_9_test.csv'

    split_ratio = 1.0
    src_train_dataset, _ = get_datasets(ds_file_name=source_train_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_train_dataset, _ = get_datasets(ds_file_name=target_train_file_name, shuffle=True, split_ratio=split_ratio)
    src_test_dataset, _ = get_datasets(ds_file_name=source_test_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_test_dataset, _ = get_datasets(ds_file_name=target_test_file_name, shuffle=True, split_ratio=split_ratio)

    # batch_size_list = [128]
    batch_size_list = [256]
    learning_rate_list = [1.2e-3]
    # learning_rate_list = [1.2e-3]
    # learning_rate_list = [6e-4]
    # batch_size_list = [256, 512]
    # learning_rate_list = [3e-4, 8e-4, 1e-3]
    tries = 1
    pos_class_weight = 1.0
    param_comb_list = list()
    for lr in learning_rate_list:
        for bs in batch_size_list:
            param_comb_list.append((lr, bs))

    date = get_current_date() + "_PDD"
    timestamp = get_timestamp()
    for param_comb in param_comb_list:
        lr, bs = param_comb
        for version in range(tries):
            task_id = date + "_pw" + str(pos_class_weight) + "_bs" + str(bs) + "_lr" + str(lr) + "_v" + str(
                version) + "_t" + str(timestamp)
            print("[INFO] perform task:{0}".format(task_id))

            global_model = create_pdd_global_model(pos_class_weight=1.0)
            print("[INFO] model created.")

            src_train_loader = get_dataloader(src_train_dataset, batch_size=bs)
            tgt_train_loader = get_dataloader(tgt_train_dataset, batch_size=bs)
            print("[INFO] train data loaded.")

            src_test_loader = get_dataloader(src_test_dataset, batch_size=bs * 2)
            tgt_test_loader = get_dataloader(tgt_test_dataset, batch_size=bs * 2)
            print("[INFO] test data loaded.")

            plat = FederatedDAANLearner(model=global_model,
                                        source_train_loader=src_train_loader,
                                        source_val_loader=src_test_loader,
                                        target_train_loader=tgt_train_loader,
                                        target_val_loader=tgt_test_loader,
                                        max_epochs=400,
                                        epoch_patience=10)
            plat.set_model_save_info(exp_dir)
            plat.train_dann(epochs=120, lr=lr, task_id=task_id)

            global_model.print_parameters()
