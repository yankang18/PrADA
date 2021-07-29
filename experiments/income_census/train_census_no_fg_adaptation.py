from collections import OrderedDict

from data_process.census_process.mapping_resource import embedding_dim_map
from experiments.income_census.global_config import pre_train_hyperparameters, data_hyperparameters, data_tag, \
    no_fg_feature_extractor_architecture
from experiments.income_census.train_census_fg_adaptation import create_embedding_dict
from experiments.income_census.train_census_utils import pretrain_census
from models.classifier import GlobalClassifier, CensusFeatureAggregator
from models.dann_models import GlobalModel, RegionalModel
from models.discriminator import LendingRegionDiscriminator
from models.feature_extractor import CensusRegionFeatureExtractorDense
from utils import compute_parameter_size


def partition_data(data):
    wide_feat = [data[:, 0].reshape(-1, 1),
                 data[:, 1].reshape(-1, 1),
                 data[:, 2].reshape(-1, 1),
                 data[:, 3].reshape(-1, 1),
                 data[:, 4].reshape(-1, 1)
                 # data[:, 5].reshape(-1, 1)
                 ]

    deep_feat = {"embeddings": OrderedDict({"class_worker": data[:, 6],
                                            "major_ind_code": data[:, 7],
                                            "major_occ_code": data[:, 8],
                                            "unemp_reason": data[:, 9],
                                            "full_or_part_emp": data[:, 10],
                                            "own_or_self": data[:, 11],
                                            "education": data[:, 12],
                                            "race": data[:, 13],
                                            "age_index": data[:, 14],
                                            "gender_index": data[:, 15],
                                            "marital_stat": data[:, 16],
                                            "union_member": data[:, 17],
                                            "vet_benefits": data[:, 18],
                                            "vet_question": data[:, 19],
                                            "region_prev_res": data[:, 20],
                                            "state_prev_res": data[:, 21],
                                            "mig_chg_msa": data[:, 22],
                                            "mig_chg_reg": data[:, 23],
                                            "mig_move_reg": data[:, 24],
                                            "mig_same": data[:, 25],
                                            "mig_prev_sunbelt": data[:, 26],
                                            "tax_filer_stat": data[:, 27],
                                            "det_hh_fam_stat": data[:, 28],
                                            "det_hh_summ": data[:, 29],
                                            "fam_under_18": data[:, 30],
                                            "hisp_origin": data[:, 31],
                                            "country_father": data[:, 32],
                                            "country_mother": data[:, 33],
                                            "country_self": data[:, 34],
                                            "citizenship": data[:, 35]
                                            })}

    deep_partition = [deep_feat]
    return wide_feat, deep_partition


def create_region_model(extractor_input_dims_list, aggregation_dim):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims_list)
    aggregator = CensusFeatureAggregator(input_dim=extractor_input_dims_list[-1], output_dim=aggregation_dim)
    discriminator = LendingRegionDiscriminator(input_dim=extractor_input_dims_list[-1], hidden_dim=36)
    return RegionalModel(extractor=extractor,
                         aggregator=aggregator,
                         discriminator=discriminator)


def create_region_model_list(feature_extractor_arch_list, aggregation_dim):
    model_list = list()
    for feature_extractor_arch in feature_extractor_arch_list:
        model_list.append(create_region_model(feature_extractor_arch, aggregation_dim))
    return model_list


def create_no_fg_census_global_model(aggregation_dim=4, num_wide_feature=5, pos_class_weight=1.0):
    embedding_dict = create_embedding_dict(embedding_dim_map)
    feature_extractor_architecture = no_fg_feature_extractor_architecture
    print(f"[INFO] # of parameter:{compute_parameter_size([feature_extractor_architecture])}")
    region_model_list = create_region_model_list([feature_extractor_architecture], aggregation_dim)

    global_input_dim = aggregation_dim + num_wide_feature
    print(f"[INFO] global_input_dim:{global_input_dim}")
    source_classifier = GlobalClassifier(input_dim=global_input_dim)
    model = GlobalModel(source_classifier=source_classifier,
                        regional_model_list=region_model_list,
                        embedding_dict=embedding_dict,
                        partition_data_fn=partition_data,
                        pos_class_weight=pos_class_weight,
                        loss_name="BCE")
    return model


if __name__ == "__main__":
    census_dann_root_dir = data_hyperparameters["census_no-fg_dann_model_dir"]
    init_model = create_no_fg_census_global_model()
    task_id = pretrain_census(data_tag,
                              census_dann_root_dir,
                              pre_train_hyperparameters,
                              data_hyperparameters,
                              init_model)
    print(f"[INFO] task id:{task_id}")
