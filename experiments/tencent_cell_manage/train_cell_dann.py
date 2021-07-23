from datasets.cell_manage_dataloader import get_dataset, get_cell_manager_dataloader
from models.classifier import CensusFeatureAggregator
from models.dann_models import create_embeddings
from models.discriminator import CensusRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from models.model_config import wire_fg_dann_global_model
from utils import get_timestamp


def partition_data(data):
    """
    partition data into party C and party A(B). Data in party C is partitioned into group
    """
    deep_partition = list()

    # in this particular Tencent Cell Data scenario, data[0] is the tabular data set located in party A (or party B)
    wide_feat = [data[0]]

    # all other tabular data sets are located in party C (each tabular data set corresponds to a feature group)
    for idx in range(1, len(data)):
        feat = {"non_embedding": {"tabular_data": data[idx]}}
        deep_partition.append(feat)

    return wide_feat, deep_partition


def create_model_group(extractor_input_dims):
    """
    create a group of models, namely feature extractor, aggregator and discriminator, for each feature group
    """
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims)
    aggregator = CensusFeatureAggregator(input_dim=extractor_input_dims[-1])
    discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dims[-1])
    return extractor, aggregator, discriminator


def create_embedding_dict():
    """
    create embedding dictionary for categorical features/columns
    """

    cat_feature_dict = {'年龄分段': 10, '年龄预测': 7, '学历': 8, '资产属性': 6, '收入水平': 6}
    embedding_meta_dict = dict()
    for col, num_values in cat_feature_dict.items():
        embedding_meta_dict[col] = (num_values, num_values - 1)
    print(f"[INFO] Create embedding_meta_dict: \t {embedding_meta_dict}")
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


def create_global_dann_model(pos_class_weight=1.0):
    # embedding_dim = 8
    embedding_dict = create_embedding_dict()

    # feature extractor architectures for feature groups (only contain dense layers)
    input_dims_list = [[17, 24, 17, 6],
                       [16, 20, 16, 6],
                       [22, 26, 22, 6],
                       [16, 20, 16, 6],
                       [100, 80, 40, 10],
                       [28, 36, 28, 6],
                       [100, 80, 40, 10],
                       [40, 50, 30, 8],
                       [52, 50, 30, 8],
                       [84, 60, 40, 10]]

    # input_dims_list = [[17, 24, 17, 8],
    #                    [16, 20, 16, 8],
    #                    [22, 26, 22, 8],
    #                    [16, 20, 16, 8],
    #                    [100, 80, 40, 8],
    #                    [28, 36, 28, 8],
    #                    [100, 80, 40, 8],
    #                    [40, 50, 30, 8],
    #                    [52, 50, 30, 8],
    #                    [84, 60, 40, 8]]

    # input_dims_list = [[17, 24, 17, 6],
    #                    [16, 20, 16, 6],
    #                    [22, 26, 22, 6],
    #                    [16, 20, 16, 6],
    #                    [100, 120, 80, 10],
    #                    [28, 36, 28, 6],
    #                    [100, 120, 80, 10],
    #                    [40, 50, 30, 8],
    #                    [52, 70, 30, 8],
    #                    [84, 100, 40, 10]]
    # architecture 2
    # input_dims_list = [[20, 25, 20, 6],
    #                    [16, 20, 16, 6],
    #                    [22, 28, 22, 6],
    #                    [16, 20, 16, 6],
    #                    [100, 120, 40, 10],
    #                    [28, 36, 28, 6],
    #                    [100, 120, 40, 10],
    #                    [40, 50, 30, 8],
    #                    [52, 60, 30, 8],
    #                    [83, 90, 40, 10]]
    # input_dims_list = [[20, 24, 16, 6],
    #                    [16, 20, 12, 6],
    #                    [22, 26, 16, 6],
    #                    [16, 20, 12, 6],
    #                    [100, 80, 40, 10],
    #                    [28, 36, 20, 6],
    #                    [100, 80, 40, 10],
    #                    [40, 50, 30, 8],
    #                    [52, 50, 30, 8],
    #                    [83, 60, 30, 10]]

    num_wide_feature = 3
    using_transform_matrix = True
    using_feature_group = True

    global_model = wire_fg_dann_global_model(embedding_dict=embedding_dict,
                                             feature_extractor_architecture_list=input_dims_list,
                                             num_wide_feature=num_wide_feature,
                                             using_feature_group=using_feature_group,
                                             using_transform_matrix=using_transform_matrix,
                                             partition_data_fn=partition_data,
                                             create_model_group_fn=create_model_group,
                                             pos_class_weight=pos_class_weight)

    return global_model


if __name__ == "__main__":

    # src_train_loader = get_cell_manager_dataloader_ob(dir=source_dir, batch_size=batch_size, data_mode="train")
    # src_test_loader = get_cell_manager_dataloader_ob(dir=source_dir, batch_size=batch_size * 2, data_mode="test")
    # tgt_train_loader = get_cell_manager_dataloader_ob(dir=target_dir, batch_size=batch_size, data_mode="train")
    # tgt_train_loader = get_cell_manager_dataloader(dir=target_dir, batch_size=batch_size, data_mode="train_bal",
    #                                                suffix="_1200_6684")
    # tgt_val_loader = get_cell_manager_dataloader(dir=target_dir, batch_size=batch_size * 2, data_mode="val")
    # tgt_test_loader = get_cell_manager_dataloader_ob(dir=target_dir, batch_size=batch_size * 2, data_mode="test")

    # load data

    # source_dir = "../../data/cell_manager/A_train_data/"
    # target_dir = "../../data/cell_manager/B_train_data/"
    # source_dir = "/Users/yankang/Documents/Data/cell_manager/A_train_data_2/"
    # target_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_2/"
    # target_dir = "/Users/yankang/Documents/Data/cell_manager/C_train_data_2/"

    source_dir = "/Users/yankang/Documents/Data/cell_manager/A_train_data_3/"
    # target_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_3/"
    target_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_4/"
    exp_dir = "cell_dann"

    print(f"[INFO] load source data from {source_dir}")
    src_train_ds = get_dataset(dir=source_dir, data_mode="train")
    src_test_ds = get_dataset(dir=source_dir, data_mode="test")

    print(f"[INFO] load target data from {target_dir}")
    tgt_train_ds = get_dataset(dir=target_dir, data_mode="train")
    tgt_test_ds = get_dataset(dir=target_dir, data_mode="test")

    print("[INFO] data loaded.")
    timestamp = get_timestamp()
    # batch_size_list = [1024]
    # batch_size_list = [512, 1024]
    # tries = 1
    # batch_size_list = [256]
    batch_size_list = [256]
    learning_rate_list = [8e-4]
    # learning_rate_list = [1.0e-3]
    # learning_rate_list = [8e-4, 1e-3, 1.2e-3]
    tries = 1
    pos_class_weight = 1.0
    epoch_patience = 1.0
    param_comb_list = list()
    # create a list of hypter-parameter combination
    for lr in learning_rate_list:
        for bs in batch_size_list:
            param_comb_list.append((lr, bs))

    date = "20210408_cell"
    for param_comb in param_comb_list:
        lr, bs = param_comb
        for version in range(tries):
            task_id = date + "_pw" + str(pos_class_weight) + "_bs" + str(bs) + "_lr" + str(lr) + "_v" + str(
                version) + "_t" + str(timestamp)
            print("[INFO] perform task:{0}".format(task_id))

            daan_model = create_global_dann_model(pos_class_weight=pos_class_weight)
            print("[INFO] model created.")

            src_train_loader = get_cell_manager_dataloader(src_train_ds, batch_size=bs)
            src_test_loader = get_cell_manager_dataloader(src_test_ds, batch_size=bs * 2)
            tgt_train_loader = get_cell_manager_dataloader(tgt_train_ds, batch_size=bs)
            tgt_test_loader = get_cell_manager_dataloader(tgt_test_ds, batch_size=bs * 2)

            plat = FederatedDAANLearner(model=daan_model,
                                        source_da_train_loader=src_train_loader,
                                        source_val_loader=src_test_loader,
                                        target_da_train_loader=tgt_train_loader,
                                        target_val_loader=tgt_test_loader,
                                        max_epochs=400,
                                        validation_batch_interval=10,
                                        epoch_patience=epoch_patience)

            # wrapper.print_global_classifier_param()
            plat.set_model_save_info(exp_dir)
            # plat.train_dann(epochs=300, lr=3e-3, task_id=task_id)
            plat.train_dann(epochs=200, src_lr=lr, task_id=task_id)
            # plat.train_source_only(epochs=200, lr=3e-3)

            daan_model.print_parameters()
