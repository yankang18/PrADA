from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census.train_census_no_fg_dann import create_global_model
from experiments.income_census.train_census_target_test import test_classifier
from models.experiment_target_learner import FederatedTargetLearner
from utils import get_timestamp, create_id_from_hyperparameters

if __name__ == "__main__":

    # TODO: all4000pos004v4
    # all4000pos004v4 # param = 34800
    # SOURCE: acc:0.9222502870264064, auc:0.922972329614549, ks:0.6940426074754433
    # TARGET: acc:0.6880924555252564, auc:0.7825527933972604, ks:0.4499684613145873
    # DA:     acc:0.6553694325412284, auc:0.8063614388756072, ks:0.4775841308649853
    # dann_task_id = '20210705_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1625434585'

    # all4000pos004v4 # param = 34800
    # SOURCE: acc:0.9254420206659013, auc:0.9301352286522875, ks:0.7102181400688864
    # TARGET: acc:0.6866640696013505, auc:0.7921722018055933, ks:0.45178117022374825
    # DA:     acc:0.6567978184651344, auc:0.8147371251299896, ks:0.4913200813502699
    # dann_task_id = '20210705_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1625436092'

    # all4000pos004v4 # param = 34800
    # SOURCE: acc:0.922870264064294, auc:0.9251231639500399, ks:0.6956244418930986
    # TARGET: acc:0.6821192052980133, auc:0.7792213648274982, ks:0.4497159914403745
    # DA: acc:0.6564082586677055, auc:0.8037118903906015, ks:0.47345997021978226
    # dann_task_id = '20210705_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1625444680'

    # all4000pos004v4 # param = 34800
    # SOURCE: acc:0.9249827784156143, auc:0.9276258526028048, ks:0.7050899349406812
    # TARGET: acc:0.6869237761329697, auc:0.7897274068989089, ks:0.4503583570371658
    # DA: acc:0.657317231528373, auc:0.8132462187113786, ks:0.4864889613142956
    # dann_task_id = '20210705_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1625449634'

    # all4000pos004v4 # param = 34800
    # SOURCE: acc:0.9257175660160735, auc:0.9309499222952621, ks:0.710549815027427
    # TARGET: acc:0.6865342163355408, auc:0.7937053931006246, ks:0.45448426740139314
    # DA:     acc:0.6562784054018959, auc:0.8172947040087812, ks:0.4881778776024883
    # dann_task_id = '20210713_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626114539'

    # TODO: all4000pos002
    # all4000pos002 # param = 34800
    # SOURCE: acc:0.9206888633754305, auc:0.9227211289661381, ks:0.6983798953948208
    # TARGET: acc:0.6873801304181051, auc:0.7807271955488441, ks:0.4557952384535696
    # DA: acc:0.6372586625751183, auc:0.7999117443708481, ks:0.4527359598653654
    # dann_task_id = '20210714_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626203544'

    # all4000pos002 # param = 34800
    # SOURCE: acc:0.9254649827784156, auc:0.9303677467705777, ks:0.7110600841944126
    # TARGET: acc:0.6949239227720241, auc:0.7898487709135924, ks:0.4596179045416252
    # DA: acc:0.6373865234624728, auc:0.8035664779391338, ks:0.46918295047500613
    # dann_task_id = '20210714_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626204106'

    # all4000pos002 # param = 34800
    # SOURCE: acc:0.9254420206659013, auc:0.931280777073741, ks:0.7158821278224263
    # TARGET: acc:0.6965861143076333, auc:0.7973672608179803, ks:0.46662969723064457
    # DA:     acc:0.6390487149980821, auc:0.8127070778765578, ks:0.47278467691607856
    # dann_task_id = '20210714_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626206636'

    # all4000pos002 # param = 34800
    # SOURCE: acc:0.9198163030998852, auc:0.9224244053950859, ks:0.6942467151422376
    # TARGET: acc:0.6831607211354047, auc:0.7833189930017862, ks:0.4581593775442714
    # DA:     acc:0.6373865234624728, auc:0.8045935573319996, ks:0.4719875543933043
    # dann_task_id = '20210714_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626208320'
    
    # all4000pos002 # param = 34800
    # SOURCE: acc:0.9246613088404133, auc:0.9287634804282335, ks:0.7115703533613982
    # TARGET: acc:0.6972254187444061, auc:0.7883304473504499, ks:0.4503833486755785
    # DA:     acc:0.6376422452371819, auc:0.8077098389149119, ks:0.4767374031868493
    # dann_task_id = '20210714_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626208798'

    # TODO: all4000pos001v2
    # all4000pos001v2 # param = 34800
    # SOURCE: acc:0.92300803673938, auc:0.9238927854764857, ks:0.6964918994769741
    # TARGET: acc:0.6823789118296325, auc:0.7733915926584123, ks:0.431795733718427
    # DA:     acc:0.6503051551746526, auc:0.7880412974516509, ks:0.42946744093677625
    # dann_task_id = '20210714_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626212684'

    # all4000pos001v2 # param = 34800
    # SOURCE: acc:0.9245924225028702, auc:0.9287446221452426, ks:0.7083811710677382
    # TARGET: acc:0.688611868588495, auc:0.7734910714754388, ks:0.42851785384130403
    # DA:     acc:0.6490066225165563, auc:0.7956247255128279, ks:0.44273621644257627
    # dann_task_id = '20210714_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626213408'

    # all4000pos001v2 # param = 34800
    # SOURCE: acc:0.925028702640643, auc:0.9284199392217283, ks:0.706620742441638
    # TARGET: acc:0.6801714063108687, auc:0.7784496293803487, ks:0.43316029584249266
    # DA:     acc:0.6470588235294118, auc:0.7904544151534998, ks:0.4317597186685105
    # dann_task_id = '20210714_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626215517'

    # all4000pos001v2 # param = 34800
    # SOURCE: acc:0.9246613088404133, auc:0.9293223159717325, ks:0.7107028957775227
    # TARGET: acc:0.6873133359303987, auc:0.7720665814747916, ks:0.4283983626129371
    # DA:     acc:0.6478379431242696, auc:0.7970721441231894, ks:0.4354411274956097
    # dann_task_id = '20210714_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626216821'

    # all4000pos001v2 # param = 34800
    # SOURCE: acc:0.9231687715269805, auc:0.9227008412264679, ks:0.6973083301441511
    # TARGET: acc:0.6823789118296325, auc:0.7715009483477111, ks:0.4278351718019948
    # DA:     acc:0.6505648617062719, auc:0.7905086928957324, ks:0.4339193093742363
    # dann_task_id = '20210714_no_fg_dann_lr0.0006_bs128_pw1.0_me600_ts1626224780'

    # Hyper-parameters
    # weight_decay = 0.00001
    weight_decay = 0.0
    batch_size = 128
    lr = 8e-4
    # lr = 5e-4
    pos_class_weight = 1.0

    timestamp = get_timestamp()
    dann_root_folder = "census_no_fg_dann"

    # Load models
    model = create_global_model(pos_class_weight=pos_class_weight,
                                aggregation_dim=4,
                                num_wide_feature=5)

    # load pre-trained model
    model.load_model(root=dann_root_folder,
                     task_id=dann_task_id,
                     load_global_classifier=False,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    # target_train_file_name = data_dir + 'grad_census9495_da_300_train.csv'
    # target_test_file_name = data_dir + 'grad_census9495_da_300_test.csv'
    # target_train_file_name = data_dir + 'grad_census9495_da_200_train.csv'
    # target_test_file_name = data_dir + 'grad_census9495_da_200_test.csv'

    # data_tag = "all4000pos004v4"
    # data_tag = "all4000pos002"
    data_tag = 'all4000pos001v2'
    target_train_file_name = data_dir + f'grad_census9495_ft_{data_tag}_train.csv'
    target_test_file_name = data_dir + f'grad_census9495_ft_{data_tag}_test.csv'

    print("[INFO] Load train data")
    target_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    target_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)

    # perform target training

    plat_target = FederatedTargetLearner(model=model,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=500)
    plat_target.set_model_save_info("census_target")

    hyperparameter_dict = {"lr": lr, "bs": batch_size, "pw": pos_class_weight, "ts": timestamp}
    appendix = create_id_from_hyperparameters(hyperparameter_dict)
    target_task_id = dann_task_id + "@target_ft" + appendix
    plat_target.train_target_with_alternating(global_epochs=500,
                                              top_epochs=1,
                                              bottom_epochs=1,
                                              lr=lr,
                                              task_id=target_task_id,
                                              dann_exp_result=None,
                                              metric=('ks', 'auc'),
                                              weight_decay=weight_decay)
    # plat_target.train_target_as_whole(global_epochs=100, lr=4e-4, task_id=target_task_id,
    #                                   dann_exp_result=dann_exp_result)

    print("[DEBUG] Global classifier Model Parameter After train:")
    model.print_parameters()

    acc, auc, ks = test_classifier(model, target_valid_loader, 'test')
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
