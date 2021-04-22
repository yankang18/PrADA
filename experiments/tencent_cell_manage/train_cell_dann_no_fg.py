import torch

from datasets.cell_manage_dataloader import get_dataset, get_cell_manager_dataloader
from models.classifier import GlobalClassifier, IdentityRegionAggregator
from models.dann_models import GlobalModel, RegionalModel, create_embeddings
from models.discriminator import CellNoFeatureGroupDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from utils import get_timestamp


def partition_data(data):
    wide_feat = list()
    agg_domain = dict({'non_embedding': dict()})
    non_embed_list = list()
    for idx in range(0, len(data)):
        non_embed_list.append(data[idx])
    agg_domain['non_embedding']['tabular_data'] = torch.cat(non_embed_list, dim=1)
    return wide_feat, [agg_domain]


def create_region_model_wrapper(extractor_input_dims_list):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims_list)
    region_aggregator = IdentityRegionAggregator()
    discriminator = CellNoFeatureGroupDiscriminator(input_dim=extractor_input_dims_list[-1])
    return RegionalModel(extractor=extractor,
                         aggregator=region_aggregator,
                         discriminator=discriminator)


# def create_region_model_wrapper(extractor_input_dims):
#     extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims)
#     region_classifier = CensusRegionAggregator(input_dim=extractor_input_dims[-1])
#     discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dims[-1])
#     return RegionalModel(extractor=extractor,
#                          aggregator=region_classifier,
#                          discriminator=discriminator)


def create_embedding_dict():
    COLUMNS = {'年龄分段': 10, '年龄预测': 7, '学历': 8, '资产属性': 6, '收入水平': 6}
    embedding_meta_dict = dict()
    for col, num_values in COLUMNS.items():
        embedding_meta_dict[col] = (num_values, num_values - 1)
    print(f"[INFO] Create embedding_meta_dict: \t {embedding_meta_dict}")
    return create_embeddings(embedding_meta_dict)


def create_region_model_wrappers(input_dims_list):
    wrapper_list = list()
    for input_dims in input_dims_list:
        wrapper_list.append(create_region_model_wrapper(input_dims))
    return wrapper_list


def create_global_daan_model(pos_class_weight=1.0):
    # embedding_dim = 8
    embedding_dict = create_embedding_dict()

    # architecture 1
    # input_dims_list = [[20, 24, 20, 6],
    #                    [16, 20, 16, 6],
    #                    [22, 26, 22, 6],
    #                    [16, 20, 16, 6],
    #                    [100, 80, 40, 10],
    #                    [28, 36, 28, 6],
    #                    [100, 80, 40, 10],
    #                    [40, 50, 30, 8],
    #                    [52, 50, 30, 8],
    #                    [84, 60, 40, 10]]
    # global_input_dim = 10
    input_dims_list = [[478, 446, 282, 72]]
    global_input_dim = 72

    region_wrapper_list = create_region_model_wrappers(input_dims_list=input_dims_list)
    beta = 1.0
    print(f"[INFO] global_input_dim:{global_input_dim}\t beta:{beta}\t pos_class_weight:{pos_class_weight}")
    classifier = GlobalClassifier(input_dim=global_input_dim)
    model = GlobalModel(classifier, region_wrapper_list, embedding_dict, partition_data_fn=partition_data,
                        beta=beta, pos_class_weight=pos_class_weight, loss_name="BCE")
    return model


if __name__ == "__main__":

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
    # batch_size_list = [256, 512, 1024]
    # learning_rate_list = [5e-4, 1e-3, 3e-3]
    # tries = 1
    # batch_size_list = [256]
    batch_size_list = [128]
    learning_rate_list = [3e-4]
    tries = 1
    pos_class_weight = 1.0
    param_comb_list = list()
    for lr in learning_rate_list:
        for bs in batch_size_list:
            param_comb_list.append((lr, bs))

    date = "20200118_cell_no_fg"
    for param_comb in param_comb_list:
        lr, bs = param_comb
        for version in range(tries):
            task_id = date + "_pw" + str(pos_class_weight) + "_bs" + str(bs) + "_lr" + str(lr) + "_v" + str(
                version) + "_t" + str(timestamp)
            print("[INFO] perform task:{0}".format(task_id))

            daan_model = create_global_daan_model(pos_class_weight=pos_class_weight)
            print("[INFO] model created.")

            src_train_loader = get_cell_manager_dataloader(src_train_ds, batch_size=bs)
            src_test_loader = get_cell_manager_dataloader(src_test_ds, batch_size=bs * 4)
            tgt_train_loader = get_cell_manager_dataloader(tgt_train_ds, batch_size=bs)
            tgt_test_loader = get_cell_manager_dataloader(tgt_test_ds, batch_size=bs * 4)

            plat = FederatedDAANLearner(model=daan_model,
                                        source_train_loader=src_train_loader,
                                        source_val_loader=src_test_loader,
                                        target_train_loader=tgt_train_loader,
                                        target_val_loader=tgt_test_loader,
                                        max_epochs=400,
                                        epoch_patience=1.2)

            # wrapper.print_global_classifier_param()
            plat.set_model_save_info(exp_dir)
            # plat.train_dann(epochs=300, lr=3e-3, task_id=task_id)
            plat.train_dann(epochs=200, lr=lr, task_id=task_id)
            # plat.train_source_only(epochs=200, lr=3e-3)

            daan_model.print_parameters()
