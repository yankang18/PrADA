from datasets.census_dataloader import get_income_census_dataloaders
from models.experiment_target_learner import FederatedTargetLearner
from experiments.income_census.train_census_dann import create_global_model_model
from experiments.income_census.train_census_target_test import test_classification

if __name__ == "__main__":
    dann_root_folder = "census_dann"
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

    # pos 200
    # apply_global_domain_adaption = True
    # SOURCE: acc:0.9586562763268744, auc:0.9173554504225361, ks:0.6777886070236744
    # TARGET: acc:0.6704024452368823, auc:0.783640103651858, ks:0.43291454926350137
    # adaption:
    # fine_tune:
    dann_task_id = '20210507_DEGREE_0.0005_64_1620343112'

    # Load models
    model = create_global_model_model(pos_class_weight=1.0)

    # load pre-trained model
    load_global_classifier = False
    model.load_model(root=dann_root_folder, task_id=dann_task_id, load_global_classifier=load_global_classifier,
                     timestamp=None)
    # dann_exp_result = load_dann_experiment_result(root=dann_root_folder, task_id=dann_task_id)
    dann_exp_result = None

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    target_train_file_name = data_dir + 'grad_census9495_da_train.csv'
    target_test_file_name = data_dir + 'grad_census9495_da_test.csv'
    # target_train_file_name = data_dir + 'grad_census9495_da_300_train.csv'
    # target_test_file_name = data_dir + 'grad_census9495_da_300_test.csv'

    # batch_size = 64; lr = 5e-4;  version = 1
    batch_size = 64; lr = 8e-4;  version = 1
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
                                         max_global_epochs=400)
    plat_target.set_model_save_info("census_target")

    appendix = "_" + str(batch_size) + "_" + str(lr) + "_v" + str(version)
    target_task_id = dann_task_id + "_target_finetune" + appendix
    plat_target.train_target_with_alternating(global_epochs=400, top_epochs=1, bottom_epochs=1, lr=lr,
                                              task_id=target_task_id, dann_exp_result=dann_exp_result)

    print("[DEBUG] Global classifier Model Parameter After train:")
    model.print_parameters()

    acc, auc, ks = test_classification(model, target_valid_loader, 'test')
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
