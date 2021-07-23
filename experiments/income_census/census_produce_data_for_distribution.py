from datasets.census_dataloader import get_census_adult_dataloaders
from experiments.income_census import train_census_fg_dann as fg_dann
from experiments.income_census import train_census_no_fg_dann as no_fg_dann
from utils import produce_data_for_distribution

if __name__ == "__main__":

    # hyper-parameters
    use_feature_group = True
    batch_size = 4000

    ############
    ##
    ############

    # feature_group_name_list = ['all_feature']
    # dann_root_folder = "census_no_fg_dann"
    # tag = 'dann_wo_fg'
    # dann_task_id = '20210522_DEGREE_no_fg_dann_lr0.0005_bs64_pw5_mep600_ts1621652044'

    ############
    ##
    ############

    feature_group_name_list = ['employment', 'demographics', 'migration', 'household']
    # feature_grp_intr_name_list = ['emp-demo', 'emp-mig', 'emp-house', 'demo-mig', 'demo-house', 'mig-house']
    # feature_group_name_list = feature_group_name_list + feature_grp_intr_name_list
    dann_root_folder = "census_dann"
    tag = 'dann'

    using_interaction = False
    aggregation_dim = 4
    # dann_task_id = '20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625414259'
    # dann_task_id = '20210704_src_lr0.0006_bs128_pw1.0_me600_ts1625369952'
    # dann_task_id = '20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625444243'
    # dann_task_id = '20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625452322'
    # dann_task_id = '20210712_src_lr0.0006_bs128_pw1.0_me600_ts1626063288'
    # dann_task_id = '20210713_src_False_lr0.0006_bs128_pw1.0_me600_ts1626161175'
    # dann_task_id = '20210713_src_lr0.0006_bs128_pw1.0_me600_ts1626112390'
    dann_task_id = '20210629_src_lr0.0006_bs128_pw1.0_me600_ts1624919877'
    version = 'ts1624919877'

    # feature group interaction
    # using_interaction = True
    # aggregation_dim = 10
    # dann_task_id = '20210712_src_lr0.0006_bs128_pw1.0_me600_ts1626060178'
    # dann_task_id = '20210706_src_lr0.0006_bs128_pw1.0_me600_ts1625473672'
    # dann_task_id = '20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625385790'
    # version = 'ts1625385790'

    ############
    ##
    ############

    # feature_group_name_list = ['employment', 'demographics', 'migration', 'household']
    # dann_root_folder = "census_no_dann"
    # tag = 'no_dann'
    #
    # using_interaction = False
    # aggregation_dim = 4
    # # dann_task_id = '20210523_income_no_da_w_fg_src_pw5_bs64_lr0.0005_ts1621734085_v0'
    # # dann_task_id = '20210524_income_no_da_w_fg_src_pw5_bs64_lr0.0005_ts1621800955_v0'
    # # dann_task_id = '20210527_income_no_da_w_fg_all_pw5_bs64_lr0.0005_ts1622052691_v0'
    # # dann_task_id = '20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626078661_v0'
    # # dann_task_id = '20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626108508_v0'
    # dann_task_id = '20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626080087_v0'
    # version = 'ts1626080087'

    ############
    ##
    ############

    # Load data
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    to_dir = "/Users/yankang/Documents/Data/census/output_emb/"

    # data_tag = "all4000pos004v4"
    # data_tag = "all4000pos001"
    # data_tag = "all4000pos001v2"
    data_tag = "all4000pos002"
    source_train_file_name = data_dir + f'undergrad_census9495_da_{data_tag}_train.csv'
    target_train_file_name = data_dir + f'grad_census9495_da_{data_tag}_train.csv'

    # source_train_file_name = data_dir + 'undergrad_census9495_da_300_train.csv'
    # target_train_file_name = data_dir + 'grad_census9495_da_300_train.csv'

    # load pre-trained model
    print("[INFO] load pre-trained model.")
    if use_feature_group:
        model = fg_dann.create_global_model(using_interaction=using_interaction)
    else:
        model = no_fg_dann.create_global_model(pos_class_weight=1,
                                               aggregation_dim=aggregation_dim,
                                               num_wide_feature=5)

    model.load_model(root=dann_root_folder,
                     task_id=dann_task_id,
                     load_global_classifier=True,
                     timestamp=None)

    print("[INFO] load training data.")
    target_train_loader, _ = get_census_adult_dataloaders(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)
    source_train_loader, _ = get_census_adult_dataloaders(
        ds_file_name=source_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] produce data for distribution.")
    produce_data_for_distribution(model,
                                  source_train_loader,
                                  target_train_loader,
                                  feature_group_name_list,
                                  to_dir,
                                  tag + str(batch_size),
                                  version=version)
