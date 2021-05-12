from datasets.ppd_dataloader import get_pdd_dataloaders_ob
from experiments.kesci_ppd.global_config import data_tag, tgt_tag
from experiments.kesci_ppd.train_ppd_fg_dann import create_pdd_global_model
from models.experiment_target_learner import FederatedTargetLearner
from utils import test_classification, get_timestamp, get_current_date

if __name__ == "__main__":
    dann_root_folder = "ppd_dann"
    # dann_task_id = '20200118_PDD_512_0.001_1'  # DA: 75.5, 39.8; TA: 75.7, 39.4
    # dann_task_id = '20200118_PDD_512_0.001_0_1610923436'  # DA: 75.6, 39.9; TA: 75.8, 40.1
    # dann_task_id = '20200118_PDD_256_0.0012_0_1610938417'  # DA: 75.9, 39.6; TA: 76.0, 39.7; TA_ft: 76.0, 40.3
    # dann_task_id = '20200118_PDD_512_0.0012_0_1610938417'  # DA: 75.7, 39.8; TA: 76.0, 40.7; TA_ft: 76.2, 41.5
    # dann_task_id = '20200119_PDD_256_0.001_0_1611043561'  # DA: 76.0, 40.7; TA:75.9, 41.4, TA_ft: 76.2, 41.8
    # dann_task_id = '20200118_PDD_512_0.001_2'  # DA: 75.8, 41.2; TA: 75.8, 41.3; TA_ft: 76.2, 42.5

    # DA: 75.7, 40.5; TA: 76.0, 42.6; 76.2, 42.5
    # dann_task_id = '20200121_PDD_pw1.0_bs512_lr0.0012_v0_t1614542852'

    # DA: 75.3, 40.8; TA: 75.7, 41.2
    # dann_task_id = '20200301_PDD_pw1.0_bs256_lr0.001_v0_t1614548613'

    # DA: 75.25, 41.0; TA: 75.4, 40.2
    # dann_task_id = '20200301_PDD_pw1.0_bs512_lr0.001_v0_t1614545680'

    # DA: 75.4, 40.8;
    # dann_task_id = '20210401_PDD_pw1.0_bs256_lr0.0015_v0_t1617224419'

    # dann_task_id = '20210401_PDD_pw1.0_bs256_lr0.0015_v0_t1617246075'

    # # DA: 75.3, 40.39;
    # dann_task_id = '20210331_PDD_pw1.0_bs256_lr0.0015_v0_t1617217008'

    # DA: 75.9, 41.33;
    # dann_task_id = '20210331_PDD_pw1.0_bs256_lr0.0015_v0_t1617157138'

    # using new data, DA: 94.28, 76.53, 41.99
    # dann_task_id = '20210501_PDD_pw1.0_bs512_lr0.001_v0_t1619854012'

    # acc:0.9401055889313672 , target auc:0.7478277836506655, target ks:0.3921640701701516
    # transfer 0.9429739017367883, auc:0.7472808537519279, ks:0.3939208930721524
    # dann_task_id = '20210506_PDD_pw1.0_bs256_lr0.0012_v0_t1620275086'

    #################
    # apply_global_domain_adaption = False
    # SOURCE: acc:0.9245929734361611, auc: 0.760931818348987, ks: 0.40209439155222293
    # TARGET: acc:0.9401055889313672, auc: 0.7410712814222382, ks: 0.3762819637471483
    # DA:  acc:0.9429739017367883, auc:0.7515160551702278, ks:0.406369998309256
    # FT:  acc:0.9429739017367883, auc:0.7516845681271977, ks:0.40270458082542315
    # dann_task_id = '20210508_PDD_pw1.0_bs256_lr0.0005_v0_t1620375550'

    # apply_global_domain_adaption = False
    # SOURCE: acc:0.9245929734361611, auc:0.7573205862792782, ks:0.40585619200077033
    # TARGET: acc:0.9401055889313672, auc:0.7418926682032966, ks:0.3779706478566837
    # DA:     acc:0.9429739017367883, auc:0.7440029118012326, ks:0.3982969363526321
    # dann_task_id = '20210508_PDD_pw1.0_bs256_lr0.0005_v0_gdFalse_t1620417860'

    # SOURCE: acc:0.9244909617660261, auc:0.8011890128660782, ks:0.4766814510790415
    # TARGET: acc:0.940196613872201, auc:0.7359031075554635, ks:0.360872794822232
    # DA:     acc:0.9429739017367883, auc:0.748923393897473, ks:0.40315437647639324
    # fine-tune: acc:0.9429739017367883, auc:0.7442585285522386, ks:0.3999427649471234
    # dann_task_id = '20210508_PDD_pw1.0_bs256_lr0.0005_v0_gdTrue_t1620419848'

    # SOURCE: acc:0.9245929734361611, auc:0.7458381764534949, ks:0.380646712725026
    # TARGET: acc:0.9401055889313672, auc:0.7420970584288233, ks:0.3798008894874264
    # DA:     acc:0.9429739017367883, auc:0.7450421418742746, ks:0.3917927051615173
    # FT:     acc:0.9429739017367883, auc:0.7451180970243225, ks:0.3923695472303078
    # dann_task_id = '20210508_PDD_pw1.0_bs256_lr0.0008_v0_gdTrue_t1620424130'

    # SOURCE: acc:0.9245929734361611, auc:0.7434543457833044, ks:0.370551887871165
    # TARGET: acc:0.9401055889313672, auc:0.7380756182031789, ks:0.3645727140667563
    # DA:     acc:0.9429739017367883, auc:0.7429122696361308, ks:0.38189431342158775
    # FT:     acc:0.9429739017367883, auc:0.745903234261724, ks:0.3925137577475055
    # dann_task_id = '20210508_PDD_pw1.0_bs64_lr0.0005_v0_gdTrue_t1620448182'

    # DA: acc:0.9429739017367883, auc:0.7563241685597973, ks:0.40244471315260877
    # dann_task_id = '20210508_PDD_pw1.0_bs64_lr0.0005_v0_gdFalse_t1620451888'

    # SOURCE: acc:0.9245929734361611, auc:0.7789747514629185, ks:0.4262767854635325
    # TARGET: acc:0.9401055889313672, auc:0.7525679740970338, ks:0.3803856603702509
    # DA:     acc:0.9429739017367883, auc:0.7562962568467914, ks:0.41273643546892164
    # FT:     acc:0.9429739017367883, auc:0.7562253546792703, ks:0.4151002688186704
    # dann_task_id = '20210508_PDD_pw1.0_bs64_lr0.0005_v0_gdFalse_t1620455269'

    # [INFO] SOURCE: acc:0.9246541804382421, auc:0.7876739105224475, ks:0.4442129155984578
    # [INFO] TARGET: acc:0.9400145639905334, auc:0.7561574578741299, ks:0.38988384631503115
    # [INFO] batch_idx:199
    # [INFO] [14/120 (26%)]
    # DA: acc:0.9429739017367883, auc:0.7644126300822176, ks:0.42722197287045666
    # dann_task_id = '20210510_PDD_pw1.0_bs64_lr0.0005_v0_gdFalse_t1620596168'

    # TODO: using data s04
    # [INFO] SOURCE: acc:0.9245929734361611, auc:0.780909689015326, ks:0.43744032072345324
    # [INFO] TARGET: acc:0.9401055889313672, auc:0.7563664097245603, ks:0.39684576881331834
    # DA: acc:0.9429739017367883, auc:0.7579339044219533, ks:0.4133623636536881
    # dann_task_id = '20210510_PDD_pw1.0_bs64_lr0.0005_v0_gdFalse_t1620621875'

    # TODO: using data s02
    # SOURCE: acc:0.9245929734361611, auc:0.7725361688894392, ks:0.41562829740540586
    # TARGET: acc:0.9401055889313672, auc:0.7481940379856807, ks:0.39360878092193086
    # DA:     acc:0.9429739017367883, auc:0.7581037808821449, ks:0.41174107811718486
    # dann_task_id = '20210511_PDD_s02_pw1.0_bs64_lr0.0005_v0_gdFalse_t1620678777'

    # SOURCE: acc:0.9245929734361611, auc:0.735470770862337, ks:0.35954758093312306
    # TARGET: acc:0.9401055889313672, auc:0.7469152379461269, ks:0.3936752923560115
    # DA:     acc:0.9429739017367883, auc:0.7531202568134255, ks:0.4198420517868148
    # dann_task_id = '20210511_PDD_s02_pw1.0_bs64_lr0.0005_v0_gdFalse_t1620709955'

    # 4000
    # [INFO] SOURCE: acc:0.9245929734361611, auc:0.7797059151405106, ks:0.4337537462537463
    # [INFO] TARGET: acc:0.9401055889313672, auc:0.7499280440458729, ks:0.38935322633429004
    # DA: acc:0.9401055889313672, auc:0.7393990780220323, ks:0.37576753017735604
    # dann_task_id = '20210512_PDD_s02_pw1.0_bs64_lr0.0005_v0_gdFalse_t1620745574'

    # [INFO] SOURCE: acc:0.9245929734361611, auc:0.7824907152037102, ks:0.43494331973247635
    # [INFO] TARGET: acc:0.9401055889313672, auc:0.7541029608771505, ks:0.38909983542834536
    dann_task_id = '20210512_PDD_s03_tgt3000_pw1.0_bs64_lr0.0005_v0_gdFalse_t1620751427'

    # hyper-parameters
    timestamp = get_timestamp()
    lr = 4e-4
    # lr = 5e-4
    # lr = 3e-4
    # lr = 2e-4
    batch_size = 64
    pos_class_weight = 1.0
    metrics =('ks', 'auc')
    load_global_classifier = False

    # Initialize models
    model = create_pdd_global_model(pos_class_weight=pos_class_weight)

    # load pre-trained model,
    model.load_model(root=dann_root_folder,
                     task_id=dann_task_id,
                     load_global_classifier=load_global_classifier,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    # data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    # target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
    # target_test_file_name = data_dir + 'PPD_2014_10to12_test.csv'

    # data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output/"
    # target_train_file_name = data_dir + 'PPD_2014_target_train.csv'
    # target_test_file_name = data_dir + 'PPD_2014_target_test.csv'

    data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_1620085151/"
    target_train_file_name = data_dir + f'PPD_2014_tgt_10to11_{data_tag}_{tgt_tag}_train.csv'
    target_test_file_name = data_dir + f'PPD_2014_tgt_10to11_{data_tag}_{tgt_tag}_test.csv'

    print("[INFO] Load train data")
    target_train_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    target_valid_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)

    # perform target training
    plat_target_dir = "pdd_target"
    plat_target = FederatedTargetLearner(model=model,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=500)
    plat_target.set_model_save_info(plat_target_dir)

    date = get_current_date() + "_PPD"
    glr = "ft_glr" if load_global_classifier else "rt_glr"
    tag = "_target_" + date + "_" + glr + "_" + str(batch_size) + "_" + str(lr) + "_v" + str(timestamp)
    target_task_id = dann_task_id + tag
    plat_target.train_target_with_alternating(global_epochs=400,
                                              top_epochs=1,
                                              bottom_epochs=1,
                                              lr=lr,
                                              task_id=target_task_id,
                                              dann_exp_result=None,
                                              metric=metrics,
                                              weight_decay=0.00001)
    # plat_target.train_target_as_whole(global_epochs=200, lr=1e-3, task_id=target_task_id,
    #                                   dann_exp_result=dann_exp_result)

    # load best model
    model.load_model(root=plat_target_dir,
                     task_id=target_task_id,
                     load_global_classifier=True,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter After train:")
    model.print_parameters()

    acc, auc, ks = test_classification(model, target_valid_loader, "test")
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
