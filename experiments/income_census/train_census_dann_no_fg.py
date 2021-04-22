from collections import OrderedDict

from datasets.census_dataloader import get_income_census_dataloaders, get_census_adult_dataloaders
from models.classifier import GlobalClassifier, CensusRegionAggregator
from models.dann_models import GlobalModel, RegionalModel, create_embeddings
from models.discriminator import CensusRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from models.discriminator import IncomeDegreeDiscriminator, LendingRegionDiscriminator
from models.classifier import GlobalClassifier, CensusRegionAggregator, IdentityRegionAggregator


def partition_data(data):
    wide_feat = [data[:, 0].reshape(-1, 1),
                 data[:, 1].reshape(-1, 1),
                 data[:, 2].reshape(-1, 1),
                 data[:, 3].reshape(-1, 1)]
    # demo_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
    #                                         "marital_status": data[:, 5],
    #                                         "gender": data[:, 6],
    #                                         "native_country": data[:, 7],
    #                                         "race": data[:, 8],
    #                                         "relationship": data[:, 12]})}
    # emp_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
    #                                        "workclass": data[:, 9],
    #                                        "occupation": data[:, 10],
    #                                        "education": data[:, 11]})}
    # demo_emp_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
    #                                             "relationship": data[:, 12],
    #                                             "occupation": data[:, 10],
    #                                             "education": data[:, 11]})}
    demo_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
                                            "marital_status": data[:, 5],
                                            "gender": data[:, 6],
                                            "native_country": data[:, 7],
                                            "race": data[:, 8],
                                            # "relationship": data[:, 12],
                                            "workclass": data[:, 9],
                                            "occupation": data[:, 10],
                                            "education": data[:, 11]
                                            })}
    deep_partition = [demo_feat]
    return wide_feat, deep_partition


# def create_region_model_wrapper(extractor_input_dim):
#     extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dim)
#     region_classifier = CensusRegionAggregator(input_dim=extractor_input_dim[-1])
#     discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dim[-1])
#     return RegionalModel(extractor=extractor,
#                          aggregator=region_classifier,
#                          discriminator=discriminator)

def create_region_model_wrapper(extractor_input_dims_list):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims_list)
    region_aggregator = IdentityRegionAggregator()
    # discriminator = IncomeDegreeDiscriminator(input_dim=extractor_input_dims_list[-1])
    discriminator = LendingRegionDiscriminator(input_dim=extractor_input_dims_list[-1])
    return RegionalModel(extractor=extractor,
                         aggregator=region_aggregator,
                         discriminator=discriminator)


def create_embedding_dict():
    feat_org2dim = {'age_bucket': 11, 'marital_status': 7, 'gender': 2, 'native_country': 43,
                    'race': 5, 'workclass': 9, 'occupation': 15, 'education': 17, 'relationship': 6}
    feat_emb2dim = {'age_bucket': 8, 'marital_status': 8, 'gender': 8, 'native_country': 8,
                    'race': 8, 'workclass': 8, 'occupation': 8, 'education': 8, 'relationship': 8}
    embedding_meta_dict = dict()
    for feat_name, num_values in feat_org2dim.items():
        embedding_dim = feat_emb2dim[feat_name]
        embedding_meta_dict[feat_name] = (num_values, embedding_dim)
    print(f"embedding_meta_dict: \n {embedding_meta_dict}")
    return create_embeddings(embedding_meta_dict)


def create_region_model_wrappers(input_dims_list):
    wrapper_list = list()
    for input_dim in input_dims_list:
        wrapper_list.append(create_region_model_wrapper(input_dim))
    return wrapper_list


def create_global_model_wrapper(pos_class_weight=1.0):
    embedding_dict = create_embedding_dict()
    # input_dims_list = [[48, 50, 48, 6],
    #                    [32, 40, 32, 6],
    #                    [32, 40, 32, 6]]
    # for RACE
    input_dims_list = [[64, 130, 64, 18]]
    # for DEGREE
    # input_dims_list = [[72, 130, 72, 18]]
    region_wrapper_list = create_region_model_wrappers(input_dims_list)

    global_input_dim = 22
    classifier = GlobalClassifier(input_dim=global_input_dim)
    wrapper = GlobalModel(classifier, region_wrapper_list, embedding_dict, partition_data,
                          pos_class_weight=pos_class_weight, loss_name="BCE")
    return wrapper


if __name__ == "__main__":
    wrapper = create_global_model_wrapper(pos_class_weight=1.0)

    # source_adult_train_file_name = '../../datasets/census_processed/degree_source_train.csv'
    # target_adult_train_file_name = '../../datasets/census_processed/degree_target_train.csv'
    # source_adult_test_file_name = '../../datasets/census_processed/degree_source_test.csv'
    # target_adult_test_file_name = '../../datasets/census_processed/degree_target_test.csv'

    source_adult_train_file_name = '../../datasets/census_processed/adult_source_train.csv'
    target_adult_train_file_name = '../../datasets/census_processed/adult_target_train.csv'
    source_adult_test_file_name = '../../datasets/census_processed/adult_source_test.csv'
    target_adult_test_file_name = '../../datasets/census_processed/adult_target_test.csv'

    # source_adult_train_file_name = '../../datasets/census_processed/standardized_adult.csv'
    # target_adult_train_file_name = '../../datasets/census_processed/sampled_standardized_census95_train.csv'
    # source_adult_test_file_name = '../../datasets/census_processed/standardized_adult_test.csv'
    # target_adult_test_file_name = '../../datasets/census_processed/sampled_standardized_census95_test.csv'
    # target_adult_test_file_name = '../../datasets/census_processed/standardized_census95_test.csv'

    batch_size = 128
    print("[INFO] Load train data")
    source_adult_train_loader, _ = get_census_adult_dataloaders(
        ds_file_name=source_adult_train_file_name, batch_size=batch_size, split_ratio=1.0)
    target_adult_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_adult_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    # census_adult_valid_loader, census_adult_test_loader = get_census_adult_dataloaders(
    #     ds_file_name=census_adult_test_file_name, batch_size=batch_size * 2, split_ratio=0.7)
    # census95_valid_loader, census95_test_loader = get_census_95_dataloaders(
    #     ds_file_name=census_95_test_file_name, batch_size=batch_size * 2, split_ratio=0.7)
    source_adult_valid_loader, _ = get_census_adult_dataloaders(
        ds_file_name=source_adult_test_file_name, batch_size=batch_size * 2, split_ratio=1.0)
    target_adult_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_adult_test_file_name, batch_size=batch_size * 2, split_ratio=1.0)

    plat = FederatedDAANLearner(model=wrapper,
                                source_train_loader=source_adult_train_loader,
                                source_val_loader=source_adult_valid_loader,
                                target_train_loader=target_adult_train_loader,
                                target_val_loader=target_adult_valid_loader,
                                max_epochs=400,
                                epoch_patience=2)

    # wrapper.print_global_classifier_param()
    plat.set_model_save_info("census_dann")
    # plat.set_model_save_info("lending_dann")
    version = 3
    lr = 1e-4
    task_id = "20201208_RACE_no_fg_" + str(lr) + "_" + str(version)
    plat.train_dann(epochs=200, lr=lr, task_id=task_id)

    wrapper.print_parameters()
