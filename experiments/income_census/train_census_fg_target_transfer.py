from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census.train_census_fg_dann import create_global_model
from experiments.income_census.train_census_target_test import test_classification
from models.experiment_target_learner import FederatedTargetLearner
from utils import get_timestamp

if __name__ == "__main__":
    # dann_task_id = "20200910_BCE_07_lr_003_w_7"
    # dann_task_id = '20201215_DEGREE_0.008_64_5'
    # dann_task_id = '20201215_DEGREE_0.008_64_6'
    # dann_task_id = '20201215_DEGREE_0.008_64_4'
    # dann_task_id = '20201215_DEGREE_0.008_64_7'
    # dann_task_id = "20201217_DEGREE_0.005_64_5"
    # dann_task_id = "20201218_DEGREE_0.008_64_2"
    # dann_task_id = "20201218_DEGREE_0.008_64_2"
    # dann_task_id = "20201218_DEGREE_0.008_64_1"

    # new
    # dann_task_id = "20210501_DEGREE_0.0008_64_1"
    # dann_task_id = "20210501_DEGREE_0.0008_64_5"
    # dann_task_id = '20210505_DEGREE_0.0008_64_2'

    # acc:0.6905247070809984 auc:0.792350433132286 ks:0.4595881559281975
    # transfer: acc:0.6826286296484972, auc:0.774328269375334, ks:0.439009453162051
    # dann_task_id = '20210505_DEGREE_0.0005_64_1620156015'

    # acc:0.6729495669893021 , target auc:0.7875587896252774, target ks:0.4606328899030785
    # transfer acc:0.6816097809475293, auc:0.766224040716187, ks:0.4315868724881645
    # dann_task_id = '20210505_DEGREE_0.0005_128_1620186352'

    ###########
    # acc:0.6877228731533367, auc:0.7814644602298758, ks:0.46029565655512406
    # transfer acc:0.6823739174732553, auc:0.7717167216897057, ks:0.45368858310001325
    # dann_task_id = '20210506_DEGREE_0.0005_64_1620240228'

    # acc:0.6938359653591442 , target auc:0.7893259613089365, target ks:0.46435337229039175
    # transfer acc:0.6811003565970454, auc:0.7713888240179085, ks:0.4365232907885027
    # dann_task_id = '20210506_DEGREE_0.0005_64_1620246052'

    # 0.6923076923076923 , target auc:0.7944239586196872, target ks:0.4649844639986166
    # transfer acc:0.6772796739684157, auc:0.7752578154690143, ks:0.4361691095976783
    # dann_task_id = '20210506_DEGREE_0.0005_64_1620247931'

    # acc:0.6793173713703515 , target auc:0.7997033552994613, target ks:0.45355414936335
    # transfer acc:0.6721854304635762, auc:0.7844211406817457, ks:0.4454226318046666
    # dann_task_id = '20210507_DEGREE_0.0005_64_1620288753'

    # source:  acc:0.9589090143218197 ,  auc:0.9248348837078167,  ks:0.7036886940201887
    # target:
    # fine_tune:
    # dann_task_id = '20210507_DEGREE_0.0005_64_1620339356'

    # pos 300
    # apply_global_domain_adaption = True
    # SOURCE: acc:0.9588037068239259, auc:0.9238804335519535, ks:0.6989307177399331
    # TARGET: acc:0.6683647478349465, auc:0.785716645280787, ks:0.44811820057570956
    # adaption: acc:0.6813550687722874, auc:0.7762812928274144, ks:0.4451184323877299
    # fine_tune: acc:0.6762419765664799, auc:0.7807023818038767, ks:0.4429907599786169
    # dann_task_id = '20210507_DEGREE_0.0005_64_1620341004'

    # pos 300
    # apply_global_domain_adaption = True
    # SOURCE: acc 0.9593407750631845, auc:0.9265234308647615, ks:0.7054857729875665
    # TARGET: acc:0.6711665817626082, auc:0.7875236013074544, ks:0.44541746127633347
    # adaption: acc:0.6711665817626082, auc:0.7875236013074544, ks:0.44541746127633347
    # fine_tune:
    # dann_task_id = '20210507_DEGREE_0.0005_128_1620357523'

    # pos 300
    # apply_global_domain_adaption = True
    # SOURCE: acc:0.9583930075821399, auc:0.9266599870712192, ks:0.7072834601892584
    # TARGET: acc:0.6683647478349465, auc:0.7979725209399215, ks:0.45791750077342486
    # adaption: acc:0.6716760061130922, auc:0.7819318185408826, ks:0.4414332819440267
    # fine_tune:
    # dann_task_id = '20210507_DEGREE_0.0005_256_1620365142'

    # pos 300
    # apply_global_domain_adaption = True
    # SOURCE: acc:0.9585404380791912, auc:0.9176335199451119, ks:0.6800891266015016
    # TARGET: acc:0.6698930208863983, auc:0.7927774325971363, ks:0.44774793329673745
    # adaption: acc:0.6709118695873663, auc:0.7846642991391931, ks:0.44192476938725506
    # fine_tune:
    # dann_task_id = '20210508_DEGREE_0.0005_64_gdTrue_1620450245'

    # [INFO] SOURCE: acc:0.9595829823083404, auc:0.9320339336458179, ks:0.7155821331423646
    # [INFO] TARGET: acc:0.6676006113092205, auc:0.7904152192777978, ks:0.44260871538510815
    # DA: acc:0.7350993377483444, auc:0.7841844453847147, ks:0.43866647478261517
    # TA: acc:0.6971472236372899, auc:0.7870377152699204, ks:0.43222399647944465
    # dann_task_id = '20210508_DEGREE_0.0005_64_gdFalse_1620457414'

    # pos_weight = 5.0
    # [INFO] SOURCE: acc:0.9352464195450716, auc:0.9318555537273328, ks:0.7137965219261184
    # [INFO] TARGET: acc:0.7320427916454406, auc:0.8115411362925415, ks:0.47372581664904373
    # DA: acc:0.7190524707080999, auc:0.7959950811040456, ks:0.45633934062550185
    # FT: acc:0.7116658176260825, auc:0.80575617684241, ks:0.4717759529642782
    # dann_task_id = '20210509_DEGREE_0.0005_64_gdFalse_1620494068'

    # DA: acc:0.7373917473255222, auc:0.7855421399495414, ks:0.4550757209511819
    # dann_task_id = '20210509_DEGREE_0.0005_64_gdFalse_1620502545'

    # [INFO] SOURCE: acc:0.9384688289806234, auc:0.9265508102789826, ks:0.7069695859400555
    # [INFO] TARGET: acc:0.7358634742740703, auc:0.8030629060839597, ks:0.46386332110503387
    # DA:            acc:0.7213448802852777, auc:0.8013043519475226, ks:0.46717533175402415
    # dann_task_id = '20210509_DEGREE_0.0005_64_gdFalse_1620512969'

    # After change embedding -----
    # "source_cls_acc": 0.9522114574557708,
    # "source_cls_auc": 0.9320826975648009,
    # "source_cls_ks": 0.7223585603372558,
    # "target_cls_acc": 0.6754966887417219,
    # "target_cls_auc": 0.7929395861106969,
    # "target_cls_ks": 0.4274880797778052,
    # DA: acc:0.7424859908303617, auc:0.7913497922740241, ks:0.4495897903896539
    # dann_task_id = '20210510_DEGREE_0.0005_64_gdFalse_1620549598'

    # TODO: 4 feature groups
    # [INFO] SOURCE: acc:0.9390901432181972, auc:0.9229817660856277, ks:0.6929378864149028
    # [INFO] TARGET: acc:0.7435048395313296, auc:0.8061511477567518, ks:0.46430195425863385
    # DA: acc:0.7514009169638308, auc:0.8012799355637269, ks:0.45973953750773067
    # dann_task_id = '20210511_DEGREE_0.0005_64_gdFalse_1620701870'

    # SOURCE: acc:0.9301811288963774, auc:0.9322626905713733, ks:0.7202765742798869
    # TARGET: acc:0.7307692307692307, auc:0.8060672702971243, ks:0.46256436948710367
    # DA: acc:0.7101375445746306, auc:0.7978847655840441, ks:0.46075411006733463
    # dann_task_id = '20210511_DEGREE_0.0005_64_gdFalse_1620714072'

    # SOURCE: acc:0.9372367312552654, auc:0.9218816362085942, ks:0.6948954885550259
    # TARGET: acc:0.7501273560876209, auc:0.7967792779012481, ks:0.4520297052597774
    # DA:     acc:0.7463066734589914, auc:0.7984407410056507, ks:0.4536805400559394
    # dann_task_id = '20210511_DEGREE_0.0005_64_gdFalse_1620718815'

    # SOURCE: acc:0.9443976411120472, auc:0.9278360562695305, ks:0.7055124212534556
    # TARGET: acc:0.7154865002547122, auc:0.8083954443049358, ks:0.4673663540507786
    # DA: acc:0.7009679062659195, auc:0.8024691570803635, ks:0.4671899815843016
    dann_task_id = '20210512_DEGREE_0.0005_64_gdFalse_1620757041'

    # Hyper-parameters

    batch_size = 128
    lr = 1e-4
    # lr = 8e-5
    # batch_size = 32; lr = 5e-4;
    # batch_size = 32; lr = 8e-4;
    # batch_size = 64; lr = 8e-4;
    # batch_size = 128; lr = 8e-4;
    pos_class_weight = 5.0
    weight_decay = 0.00001
    load_global_classifier = False

    timestamp = get_timestamp()

    dann_root_folder = "census_dann"

    # Load models
    model = create_global_model(pos_class_weight=pos_class_weight)

    # load pre-trained model
    model.load_model(root=dann_root_folder,
                     task_id=dann_task_id,
                     load_global_classifier=load_global_classifier,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    target_train_file_name = data_dir + 'grad_census9495_da_300_train.csv'
    target_test_file_name = data_dir + 'grad_census9495_da_300_test.csv'
    # target_train_file_name = data_dir + 'grad_census9495_da_200_train.csv'
    # target_test_file_name = data_dir + 'grad_census9495_da_200_test.csv'

    print("[INFO] Load train data")
    target_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    target_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)

    # perform target training
    plat_target_dir = "census_target"
    plat_target = FederatedTargetLearner(model=model,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=500)
    plat_target.set_model_save_info("census_target")

    appendix = "_" + str(batch_size) + "_" + str(lr) + "_v" + str(timestamp)
    target_task_id = dann_task_id + "_target_finetune" + appendix
    plat_target.train_target_with_alternating(global_epochs=500,
                                              top_epochs=1,
                                              bottom_epochs=1,
                                              lr=lr,
                                              task_id=target_task_id,
                                              dann_exp_result=None,
                                              metric=('ks', 'auc'),
                                              weight_decay=weight_decay)

    # load best model
    model.load_model(root=plat_target_dir,
                     task_id=target_task_id,
                     load_global_classifier=True,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter After train:")
    model.print_parameters()

    acc, auc, ks = test_classification(model, target_valid_loader, 'test')
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
