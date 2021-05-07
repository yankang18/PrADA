from models.classifier import GlobalClassifier
from models.dann_models import GlobalModel, RegionalModel
from models.discriminator import GlobalDiscriminator, CensusRegionDiscriminator
from models.interaction_models import initialize_transform_matrix_dict, AttentiveFeatureComputer, InteractionModel


def create_interaction_model(input_dims_list, create_model_group_fn, using_transform_matrix=True):
    """
    create interaction model that is responsible for interations among feature groups.
    """
    hidden_dim_list = [f[-1] for f in input_dims_list]
    if using_transform_matrix:
        print("[INFO] create transform matrix with hidden_dim_list:", hidden_dim_list)
        transform_matrix_dict = initialize_transform_matrix_dict(hidden_dim_list)
    else:
        if len(set(hidden_dim_list)) > 1:
            raise RuntimeError(f"[ERROR] all hidden dim should be the same if do not use tranform matrix,"
                               f" but get hidden_dim_list:{hidden_dim_list}")
        transform_matrix_dict = None
    att_feat_computer = AttentiveFeatureComputer(transform_matrix_dict=transform_matrix_dict)
    extractor_list, classifier_list, discriminator_list = create_model_group_list(input_dims_list,
                                                                                  create_model_group_fn)
    return InteractionModel(extractor_list=extractor_list,
                            aggregator_list=classifier_list,
                            discriminator_list=discriminator_list,
                            interactive_feature_computer=att_feat_computer)


def create_model_group_list(input_dims_list, create_model_group_fn):
    """
    create models for each interactive feature group
    """
    extractor_list = list()
    classifier_list = list()
    discriminator_list = list()
    for input_dim in input_dims_list:
        models = create_model_group_fn(input_dim)
        extractor_list.append(models[0])
        classifier_list.append(models[1])
        discriminator_list.append(models[2])
    return extractor_list, classifier_list, discriminator_list


def create_region_model(extractor_input_dim, create_model_group_fn):
    """
    create models, namely feature extractor, aggregator and discriminator, for a region representing a feature group.
    """
    extractor, aggregator, discriminator = create_model_group_fn(extractor_input_dim)
    return RegionalModel(extractor=extractor, aggregator=aggregator, discriminator=discriminator)


def create_region_model_list(input_dims_list, create_model_group_fn):
    """
    create models for all regions that each represents a feature group.
    """
    wrapper_list = list()
    for input_dim in input_dims_list:
        wrapper_list.append(create_region_model(input_dim, create_model_group_fn))
    return wrapper_list


def wire_global_model(embedding_dict,
                      input_dims_list,
                      num_wide_feature,
                      using_feature_group,
                      using_interaction,
                      using_transform_matrix,
                      partition_data_fn,
                      create_model_group_fn,
                      pos_class_weight=1.0):
    """
    wire up all models together as a single model for end-to-end training

    parameters:
    ----------
    embedding_dict - the embedding dictionary,
    input_dims_list - the neural network architecture for all feature groups (neural network has only dense layers),
    num_wide_feature - the number of feature used in party A or party B,
    using_feature_group - whether apply feature group,
    using_interaction -  whether apply interactions among feature groups,
    using_transform_matrix - whether use transform matrix when feature group interaction applied,
    partition_data_fn - the data partition function,
    create_model_group_fn - the create model group function,
    """

    if using_feature_group:
        print(f"[INFO] input_dims_list:{input_dims_list}, len:{len(input_dims_list)}")
        region_model_list = create_region_model_list(input_dims_list, create_model_group_fn)
    else:
        region_model_list = list()
    print(f"[INFO] region_model_list len:{len(region_model_list)}")

    interaction_model = None
    interactive_group_num = 0
    if using_interaction:
        interaction_model = create_interaction_model(input_dims_list, create_model_group_fn, using_transform_matrix)
        interactive_group_num = interaction_model.get_num_feature_groups()

    global_discriminator_dim = len(region_model_list) + interactive_group_num
    global_input_dim = num_wide_feature + len(region_model_list) + interactive_group_num

    print(f"[INFO] global_input_dim length:{global_input_dim}")
    print(f"[INFO] global_discriminator_dim length:{global_discriminator_dim}")

    global_classifier = GlobalClassifier(input_dim=global_input_dim)
    global_discriminator = GlobalDiscriminator(input_dim=global_discriminator_dim)
    global_model = GlobalModel(global_classifier, region_model_list, embedding_dict, partition_data_fn,
                               feature_interactive_model=interaction_model,
                               pos_class_weight=pos_class_weight,
                               loss_name="BCE",
                               discriminator=global_discriminator)
    return global_model
