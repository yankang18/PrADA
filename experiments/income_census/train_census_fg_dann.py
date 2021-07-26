from collections import OrderedDict

from data_process.census_process.mapping_resource import embedding_dim_map
from experiments.income_census.global_config import fg_feature_extractor_architecture_list, data_tag, \
    pre_train_fg_dann_hyperparameters, data_hyperparameters, intr_fg_feature_extractor_for_architecture_list
from experiments.income_census.train_census_utils import pretrain_census_dann
from models.classifier import CensusFeatureAggregator
from models.dann_models import create_embedding
from models.discriminator import CensusRegionDiscriminator
from models.feature_extractor import CensusRegionFeatureExtractorDense
from models.model_config import wire_fg_dann_global_model


def partition_data(data):
    wide_feat = [data[:, 0].reshape(-1, 1),
                 data[:, 1].reshape(-1, 1),
                 data[:, 2].reshape(-1, 1),
                 data[:, 3].reshape(-1, 1),
                 data[:, 4].reshape(-1, 1)
                 # data[:, 5].reshape(-1, 1)
                 ]

    emp_feat = {"embeddings": OrderedDict({"class_worker": data[:, 6],
                                           "major_ind_code": data[:, 7],
                                           "major_occ_code": data[:, 8],
                                           "unemp_reason": data[:, 9],
                                           "full_or_part_emp": data[:, 10],
                                           "own_or_self": data[:, 11]})}
    demo_feat = {"embeddings": OrderedDict({"education": data[:, 12],
                                            "race": data[:, 13],
                                            "age_index": data[:, 14],
                                            "gender_index": data[:, 15],
                                            "marital_stat": data[:, 16],
                                            "union_member": data[:, 17],
                                            "vet_benefits": data[:, 18],
                                            "vet_question": data[:, 19]})}
    residence_feat = {"embeddings": OrderedDict({"region_prev_res": data[:, 20],
                                                 "state_prev_res": data[:, 21],
                                                 "mig_chg_msa": data[:, 22],
                                                 "mig_chg_reg": data[:, 23],
                                                 "mig_move_reg": data[:, 24],
                                                 "mig_same": data[:, 25],
                                                 "mig_prev_sunbelt": data[:, 26],
                                                 "hisp_origin": data[:, 31],
                                                 "country_father": data[:, 32],
                                                 "country_mother": data[:, 33],
                                                 "country_self": data[:, 34],
                                                 "citizenship": data[:, 35]
                                                 })}
    household_feat = {"embeddings": OrderedDict({"tax_filer_stat": data[:, 27],
                                                 "det_hh_fam_stat": data[:, 28],
                                                 "det_hh_summ": data[:, 29],
                                                 "fam_under_18": data[:, 30]})}

    deep_partition = [emp_feat, demo_feat, residence_feat, household_feat]
    return wide_feat, deep_partition


def create_model_group(extractor_input_dim):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dim)
    aggregator = CensusFeatureAggregator(input_dim=extractor_input_dim[-1])
    discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dim[-1])
    return extractor, aggregator, discriminator


def create_embedding_dict(embedding_dim_map):
    tag_embedding_map = dict()
    feat_embedding_dict = dict()
    for feat_name, val in embedding_dim_map.items():
        # tag embedding map is used for embedding reuse. features with the same tag use the same embedding.
        tag = val[2]
        embedding = tag_embedding_map.get(tag)
        if embedding is None:
            embedding = create_embedding((val[0], val[1]))
            tag_embedding_map[tag] = embedding
        feat_embedding_dict[feat_name] = embedding
    return feat_embedding_dict


def create_global_model(pos_class_weight=1.0, num_wide_feature=5, using_interaction=False):
    embedding_dict = create_embedding_dict(embedding_dim_map)

    using_feature_group = True
    using_interaction = using_interaction
    using_transform_matrix = False

    global_model = wire_fg_dann_global_model(embedding_dict=embedding_dict,
                                             feat_extr_archit_list=fg_feature_extractor_architecture_list,
                                             intr_feat_extr_archit_list=intr_fg_feature_extractor_for_architecture_list,
                                             num_wide_feature=num_wide_feature,
                                             using_feature_group=using_feature_group,
                                             using_interaction=using_interaction,
                                             using_transform_matrix=using_transform_matrix,
                                             partition_data_fn=partition_data,
                                             create_model_group_fn=create_model_group,
                                             pos_class_weight=pos_class_weight)

    return global_model


# def train_census_dann(census_dann_root_dir,
#                       dann_hyperparameters,
#                       data_hyperparameters,
#                       create_global_model_func):
#
#     using_interaction = dann_hyperparameters['using_interaction']
#     momentum = dann_hyperparameters['momentum']
#     weight_decay = dann_hyperparameters['weight_decay']
#     batch_size = dann_hyperparameters['batch_size']
#     lr = dann_hyperparameters['lr']
#     epoch_patience = dann_hyperparameters['epoch_patience']
#     max_epochs = dann_hyperparameters['max_epochs']
#     valid_metric = dann_hyperparameters['valid_metric']
#
#     data_tag = data_hyperparameters['data_tag']
#
#     date = get_current_date()
#     timestamp = get_timestamp()
#
#     optimizer_param_dict = {"src": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay},
#                             "tgt": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}}
#
#     hyperparameter_dict = {"lr": lr, "bs": batch_size, "me": max_epochs, "ts": timestamp}
#     using_intr_tag = "intr" + str(True) if using_interaction else str(False)
#     task_id = date + "_src_" + using_intr_tag + "_" + create_id_from_hyperparameters(hyperparameter_dict)
#
#     # load data
#     print(f"[INFO] data_tag:{data_tag}.")
#     source_train_file_name = data_hyperparameters['source_train_file_name']
#     target_train_file_name = data_hyperparameters['target_train_file_name']
#     source_test_file_name = data_hyperparameters['source_test_file_name']
#     target_test_file_name = data_hyperparameters['target_test_file_name']
#
#     print("[INFO] Load train data.")
#     source_da_census_train_loader, _ = get_income_census_dataloaders(
#         ds_file_name=source_train_file_name, batch_size=batch_size, split_ratio=1.0)
#     target_da_census_train_loader, _ = get_income_census_dataloaders(
#         ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)
#     target_classifier_census_train_loader, _ = get_income_census_dataloaders(
#         ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)
#
#     print("[INFO] Load test data.")
#     source_census_valid_loader, _ = get_income_census_dataloaders(
#         ds_file_name=source_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)
#     target_census_valid_loader, _ = get_income_census_dataloaders(
#         ds_file_name=target_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)
#
#     model = create_global_model_func(num_wide_feature=5, using_interaction=using_interaction)
#     plat = FederatedDAANLearner(model=model,
#                                 source_da_train_loader=source_da_census_train_loader,
#                                 source_val_loader=source_census_valid_loader,
#                                 target_da_train_loader=target_da_census_train_loader,
#                                 target_clz_train_loader=target_classifier_census_train_loader,
#                                 target_val_loader=target_census_valid_loader,
#                                 max_epochs=max_epochs,
#                                 epoch_patience=epoch_patience)
#     plat.set_model_save_info(census_dann_root_dir)
#
#     plat.train_dann(epochs=200,
#                     task_id=task_id,
#                     metric=valid_metric,
#                     optimizer_param_dict=optimizer_param_dict)


if __name__ == "__main__":
    census_dann_root_dir = data_hyperparameters['census_fg_dann_model_dir']
    pretrain_census_dann(data_tag,
                         census_dann_root_dir,
                         pre_train_fg_dann_hyperparameters,
                         data_hyperparameters,
                         create_census_global_model_func=create_global_model)
