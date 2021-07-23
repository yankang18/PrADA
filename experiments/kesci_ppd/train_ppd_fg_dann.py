from collections import OrderedDict

from datasets.ppd_dataloader import get_datasets, get_dataloader
from experiments.kesci_ppd.global_config import feature_extractor_architecture_list, tgt_data_tag, tgt_tag, data_tag
from experiments.kesci_ppd.meta_data import column_name_list, group_ind_list, group_info, embedding_shape_map
from models.classifier import CensusFeatureAggregator
from models.dann_models import create_embeddings
from models.discriminator import CensusRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from models.model_config import wire_fg_dann_global_model
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
    classifier = CensusFeatureAggregator(input_dim=extractor_input_dim[-1])
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


def create_pdd_global_model(pos_class_weight=1.0, num_wide_feature=6, using_interaction=False):
    embedding_dict = create_embedding_dict()
    print("[INFO] embedding_dict", embedding_dict)

    num_wide_feature = num_wide_feature
    # num_wide_feature = 17
    using_feature_group = True
    using_interaction = using_interaction
    using_transform_matrix = False

    global_model = wire_fg_dann_global_model(embedding_dict=embedding_dict,
                                             feature_extractor_architecture_list=feature_extractor_architecture_list,
                                             intr_feature_extractor_architecture_list=None,
                                             num_wide_feature=num_wide_feature,
                                             using_feature_group=using_feature_group,
                                             using_interaction=using_interaction,
                                             using_transform_matrix=using_transform_matrix,
                                             partition_data_fn=partition_data,
                                             create_model_group_fn=create_model_group,
                                             pos_class_weight=pos_class_weight)

    return global_model


if __name__ == "__main__":

    ########
    # hyper-parameters
    ########

    # learning rate
    momentum = 0.99
    # weight_decay = 0.00001
    weight_decay = 0.0

    tgt_lr = 5e-4
    src_learning_rate_list = [5e-4]

    # batch size
    all_batch_size = 64
    src_batch_size_list = [all_batch_size]
    tgt_batch_size = all_batch_size

    # scenario control
    using_interaction = True
    is_all = False  # if true using src+tgt else just using src for training source classifier
    use_target_classifier = False  # apply target classifier for multi-task learning
    monitor_source = False  # monitor on source test data for validation
    apply_global_domain_adaption = False
    global_domain_adaption_lambda = 1.0
    num_tgt_clz_train_iter = 1
    tgt_clz_interval = 1
    # num_tgt_clz_train_iter = (int(3000 / tgt_batch_size) + 1) * 10

    epoch_patience = 5

    # controls alpha
    # max_epochs = 800
    max_epochs = 600
    # max_epochs = 400

    pos_class_weight = 3.0
    metrics = ('ks', 'auc')
    exp_dir = "ppd_dann"

    # load data
    # ts = '1620085151'
    # data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{ts}/"
    # source_train_file_name = data_dir + f"PPD_2014_src_1to9_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_train.csv"
    # source_test_file_name = data_dir + f'PPD_2014_src_1to9_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_test.csv'
    # target_train_file_name = data_dir + f'PPD_2014_tgt_10to11_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_train.csv'
    # target_test_file_name = data_dir + f'PPD_2014_tgt_10to11_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_test.csv'
    # src_tgt_train_file_name = data_dir + f'PPD_2014_src_tgt_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_train.csv'

    ts = '20210522'
    data_tag = 'lbl004tgt4000v4'
    data_tag = 'lbl002tgt4000'
    # data_tag = 'lbl001tgt4000'
    data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{ts}/"

    # source_train_file_name = data_dir + f"PPD_2014_src_1to9_da_{tgt_data_tag}_{tgt_tag}_train.csv"
    # source_test_file_name = data_dir + f'PPD_2014_src_1to9_da_{tgt_data_tag}_{tgt_tag}_test.csv'
    # source_train_file_name = data_dir + f"PPD_2014_src_1to8_da_{tgt_data_tag}_{tgt_tag}_train.csv"
    # source_test_file_name = data_dir + f'PPD_2014_src_1to8_da_{tgt_data_tag}_{tgt_tag}_test.csv'

    source_train_file_name = data_dir + f"PPD_2014_src_1to9_da_{data_tag}_train.csv"
    source_test_file_name = data_dir + f'PPD_2014_src_1to9_da_{data_tag}_test.csv'
    target_train_file_name = data_dir + f'PPD_2014_tgt_10to12_da_{data_tag}_train.csv'
    target_test_file_name = data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_test.csv'

    print(f"load source train from: {source_train_file_name}.")
    print(f"load target train from: {target_train_file_name}.")
    print(f"load source test from: {source_test_file_name}.")
    print(f"load target test from: {target_test_file_name}.")

    split_ratio = 1.0
    # src_tgt_train_dataset, _ = get_datasets(ds_file_name=src_tgt_train_file_name, shuffle=True, split_ratio=split_ratio)
    src_train_dataset, _ = get_datasets(ds_file_name=source_train_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_train_dataset, _ = get_datasets(ds_file_name=target_train_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_clz_train_dataset, _ = get_datasets(ds_file_name=target_train_file_name, shuffle=True, split_ratio=split_ratio)
    src_test_dataset, _ = get_datasets(ds_file_name=source_test_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_test_dataset, _ = get_datasets(ds_file_name=target_test_file_name, shuffle=True, split_ratio=split_ratio)

    tries = 1
    param_comb_list = list()
    for lr in src_learning_rate_list:
        for bs in src_batch_size_list:
            param_comb_list.append((lr, bs))

    date = get_current_date() + "_PPD_fg_dann"
    timestamp = get_timestamp()
    for param_comb in param_comb_list:
        lr, bs = param_comb
        for version in range(tries):
            train_data_tag = "all" if is_all else "src"
            using_intr_tag = "intr" + str(True) if using_interaction else str(False)
            task_id = date + "_" + train_data_tag + "_" + data_tag + "_" + using_intr_tag + "_pw" + str(
                pos_class_weight) + "_bs" + str(bs) + "_lr" + str(lr) + "_gd" + str(apply_global_domain_adaption) \
                      + "_mep" + str(max_epochs) + "_ts" + str(timestamp) + "_ve" + str(version)
            print("[INFO] perform task:{0}".format(task_id))

            global_model = create_pdd_global_model(pos_class_weight=pos_class_weight,
                                                   using_interaction=using_interaction)
            print("[INFO] model created.")

            if is_all:
                # src_train_loader = get_dataloader(src_tgt_train_dataset, batch_size=bs)
                pass
            else:
                src_train_loader = get_dataloader(src_train_dataset, batch_size=bs)
            tgt_train_loader = get_dataloader(tgt_train_dataset, batch_size=bs)
            target_clz_train_loader = get_dataloader(tgt_clz_train_dataset, batch_size=tgt_batch_size)
            print("[INFO] train data loaded.")

            src_test_loader = get_dataloader(src_test_dataset, batch_size=bs * 4)
            tgt_test_loader = get_dataloader(tgt_test_dataset, batch_size=bs * 4)
            print("[INFO] test data loaded.")

            plat = FederatedDAANLearner(model=global_model,
                                        source_da_train_loader=src_train_loader,
                                        source_val_loader=src_test_loader,
                                        target_da_train_loader=tgt_train_loader,
                                        target_val_loader=tgt_test_loader,
                                        target_clz_train_loader=target_clz_train_loader,
                                        max_epochs=max_epochs,
                                        epoch_patience=epoch_patience)
            plat.set_model_save_info(exp_dir)

            optimizer_param_dict = {"src": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay},
                                    "tgt": {"lr": tgt_lr, "momentum": momentum, "weight_decay": weight_decay}}
            plat.train_dann(epochs=120,
                            task_id=task_id,
                            metric=metrics,
                            apply_global_domain_adaption=apply_global_domain_adaption,
                            global_domain_adaption_lambda=global_domain_adaption_lambda,
                            use_target_classifier=use_target_classifier,
                            optimizer_param_dict=optimizer_param_dict,
                            monitor_source=monitor_source,
                            num_tgt_clz_train_iter=num_tgt_clz_train_iter,
                            tgt_clz_interval=tgt_clz_interval)

            global_model.print_parameters()
