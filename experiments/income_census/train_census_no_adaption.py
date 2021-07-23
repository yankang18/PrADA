from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census import train_census_fg_dann as fg_dann
from experiments.income_census import train_census_no_fg_dann as no_fg_dann
from models.experiment_dann_learner import FederatedDAANLearner
from utils import get_timestamp, get_current_date

if __name__ == "__main__":

    # TODO: all4000pos004v4
    # all4000pos004v4, src+tgt, # param = 20235
    # acc:0.673678743020387, auc:0.8037038343925939, ks:0.4753626438487753
    # 20210522_income_no_da_fg_all_pw5_bs64_lr0.0005_ts1621654962_v0

    # all4000pos004v4, src+tgt, # param = 20235
    # acc:0.6758862485391508, auc:0.8016872832171034, ks:0.47595280949467533
    # 20210712_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626074725_v0

    # all4000pos004v4, src+tgt, # param = 20235
    # acc:0.6651084274769511, auc:0.8052062597947813, ks:0.47002038714283134
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626078661_v0

    # all4000pos004v4, src+tgt, # param = 20235
    # acc:0.6705622646409557, auc:0.8087690522530256, ks:0.4836854011026729
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626080087_v0

    # all4000pos004v4, src+tgt, # param = 20235
    # acc:0.6670562264640956, auc:0.808669354721121, ks:0.47597613908166575
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626108508_v0

    # all4000pos004v4, tgt, # param = 20235
    # acc:0.6519932476301779, auc:0.7796239460130028, ks:0.43921206361278486
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626144976_v0

    # all4000pos004v4, tgt, # param = 20235
    # acc:0.6519932476301779, auc:0.7914771623756751, ks:0.4581752267781665
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626139511_v0

    # all4000pos004v4, tgt, # param = 20235
    # acc:0.6543306064147514, auc:0.7949304151325033, ks:0.4586127294391975
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626141052_v0

    # all4000pos004v4, tgt, # param = 20235
    # acc:0.6543306064147514, auc:0.7974299616636561, ks:0.46434757935121584
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626141495_v0

    # all4000pos004v4, tgt, # param = 20235
    # acc:0.6543306064147514, auc:0.7931993962302886, ks:0.4511172247589908
    # 20210714_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626203754_v0

    # TODO: all4000pos002
    # all4000pos002, src+tgt, # param = 20235
    # acc:0.6579721263265567, auc:0.7868705412483854, ks:0.45696599226796103
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626119434_v0

    # all4000pos002, src+tgt, # param = 20235
    # acc:0.6624472573839663, auc:0.7908874489306917, ks:0.4598725792115312
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626121510_v0

    # all4000pos002, src+tgt, # param = 20235
    # acc:0.6591228743127477, auc:0.7980433757428071, ks:0.4669061546826475
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626122604_v0

    # all4000pos002, src+tgt, # param = 20235
    # acc:0.6738268763585219, auc:0.7867348232001905, ks:0.45268737294555833
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626124859_v0

    # all4000pos002, src+tgt, # param = 20235
    # acc:0.658739291650684, auc:0.7862952070524161, ks:0.4594823986633858
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626125847_v0

    # all4000pos002, tgt, # param = 20235
    # acc: acc:0.6372586625751183, auc:0.7844518744791228, ks:0.42554185377699294
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626146931_v0

    # all4000pos002, tgt, # param = 20235
    # acc:0.6372586625751183, auc:0.774308417885418, ks:0.42178326301549524
    # 20210714_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626204773_v0

    # all4000pos002, tgt, # param = 20235
    # acc:0.6372586625751183, auc:0.7729988695584771, ks:0.42478659945876857
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626148296_v0

    # all4000pos002, tgt, # param = 20235
    # acc:0.6372586625751183, auc:0.7780830982018738, ks:0.4257093973185112
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626150977_v0

    # all4000pos002, tgt, # param = 20235
    # acc:0.6372586625751183, auc:0.768204146819346, ks:0.4185777993279588
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626149515_v0

    # TODO: all4000pos001
    # all4000pos001v2, src+tgt, # param = 20235
    # acc:0.656927671730944, auc:0.7740033017197988, ks:0.4237640130622357
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626129151_v0

    # all4000pos001v2, src+tgt, # param = 20235
    # acc:0.6558888456044669, auc:0.7832255968145782, ks:0.435778823267297
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626130731_v0

    # all4000pos001v2, src+tgt, # param = 20235
    # acc:0.6718607972990521, auc:0.773967104407484, ks:0.4357923106847759
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626132495_v0

    # all4000pos001v2, src+tgt, # param = 20235
    # acc:0.6562784054018959, auc:0.7750435825846966, ks:0.4240118899240097
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626133136_v0

    # all4000pos001v2, src+tgt, # param = 20235
    # acc:0.6721205038306713, auc:0.7764945006331067, ks:0.43546620680162446
    # 20210713_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626134410_v0

    # all4000pos001v2, tgt, # param = 20235
    # acc:0.6368004155304506, auc:0.7595499664345567, ks:0.4011975222812137
    # 20210714_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626205322_v0

    # all4000pos001v2, tgt, # param = 20235
    # acc:0.6368004155304506, auc:0.7639187232183633, ks:0.40432733218590644
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626159034_v0

    # all4000pos001v2, tgt, # param = 20235
    # acc:0.6430333722893131, auc:0.7684237758236656, ks:0.4106099899624452
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626154801_v0

    # all4000pos001v2, tgt, # param = 20235
    # acc:0.6451110245422672, auc:0.7680541841322647, ks:0.4125477308768834
    # 20210713_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626155391_v0

    # all4000pos001v2, tgt, # param = 20235
    # acc:0.6436826386183613, auc:0.7696134024977822, ks:0.4185977489281513
    # 20210714_income_no_da_w_fg_tgt_pw1.0_bs128_lr0.0005_ts1626206599_v0

    # TODO: all4000pos001
    # acc:0.6539105785336942, auc:0.7815253235958748, ks:0.43777550310010993
    # 20210716_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626391034_v0

    # acc:0.6549413735343383, auc:0.7858529912011332, ks:0.4388486512195374
    # 20210716_income_no_da_w_fg_all_pw1.0_bs128_lr0.0005_ts1626392241_v0

    # data_tag = "all4000pos001"
    data_tag = "all4000pos002"
    # data_tag = "all4000pos004v4"
    exp_dir = "census_no_dann"
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    # source_train_file_name = data_dir + 'undergrad_census9495_da_300_train.csv'
    # source_test_file_name = data_dir + 'undergrad_census9495_da_300_test.csv'
    # target_train_file_name = data_dir + 'grad_census9495_da_300_train.csv'
    # target_test_file_name = data_dir + 'grad_census9495_da_300_test.csv'
    src_tgt_train_file_name = data_dir + f'degree_src_tgt_census9495_da_{data_tag}_train.csv'

    source_train_file_name = data_dir + f'undergrad_census9495_da_{data_tag}_train.csv'
    target_train_file_name = data_dir + f'grad_census9495_ft_{data_tag}_train.csv'
    source_test_file_name = data_dir + f'undergrad_census9495_da_{data_tag}_test.csv'
    target_test_file_name = data_dir + f'grad_census9495_ft_{data_tag}_test.csv'

    data_file_name_dict = {"src": source_train_file_name,
                           "tgt": target_train_file_name,
                           "all": src_tgt_train_file_name}
    is_train_source = True
    is_valid_source = False
    # train_data_tag = 'src' if is_train_source else 'tgt'
    train_data_tag = 'all'  # 'all', 'src', 'tgt'
    apply_feature_group = False

    momentum = 0.99
    weight_decay = 0.00001
    batch_size_list = [128]
    # learning_rate_list = [8e-4]
    learning_rate_list = [5e-4]
    # is_all_list = [False]
    tries = 1
    pos_class_weight = 1.0
    epoch_patience = 8
    epochs = 800
    metrics = ('ks', 'auc')

    param_comb_list = list()
    for lr in learning_rate_list:
        for bs in batch_size_list:
            param_comb_list.append((lr, bs))

    timestamp = get_timestamp()
    date = get_current_date() + "_" + data_tag + "_income_no_da_w_fg" if apply_feature_group else "_income_no_da_wo_fg"
    for param_comb in param_comb_list:
        lr, bs = param_comb
        for version in range(tries):
            task_id = date + "_" + train_data_tag + "_pw" + str(pos_class_weight) + "_bs" + str(bs) + "_lr" + str(
                lr) + "_ts" + str(timestamp) + "_v" + str(version)
            print("[INFO] perform task:{0}".format(task_id))

            if apply_feature_group:
                print("[INFO] apply feature grouping")
                model = fg_dann.create_global_model(pos_class_weight=pos_class_weight,
                                                    num_wide_feature=5)
            else:
                model = no_fg_dann.create_global_model(aggregation_dim=4,
                                                       num_wide_feature=5,
                                                       pos_class_weight=pos_class_weight)
            print("[INFO] model created.")
            src_train_loader, _ = get_income_census_dataloaders(
                ds_file_name=data_file_name_dict[train_data_tag], batch_size=bs, split_ratio=1.0)
            tgt_train_loader, _ = get_income_census_dataloaders(
                ds_file_name=data_file_name_dict['tgt'], batch_size=bs, split_ratio=1.0)

            src_valid_loader, _ = get_income_census_dataloaders(
                ds_file_name=source_test_file_name, batch_size=bs * 4, split_ratio=1.0)
            tgt_valid_loader, _ = get_income_census_dataloaders(
                ds_file_name=target_test_file_name, batch_size=bs * 4, split_ratio=1.0)

            plat = FederatedDAANLearner(model=model,
                                        source_da_train_loader=src_train_loader,
                                        source_val_loader=src_valid_loader,
                                        target_da_train_loader=tgt_train_loader,
                                        target_val_loader=tgt_valid_loader,
                                        epoch_patience=epoch_patience,
                                        validation_batch_interval=5)
            plat.set_model_save_info(exp_dir)

            # optimizer_param_dict = {"src": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay},
            #                         "tgt": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}}
            plat.train_wo_adaption(epochs=epochs,
                                   lr=lr,
                                   train_source=is_train_source,
                                   valid_source=is_valid_source,
                                   metric=metrics,
                                   task_id=task_id)
