from datasets.ppd_dataloader import get_pdd_dataloaders_ob
from experiments.kesci_ppd.global_config import data_tag, tgt_tag
from experiments.kesci_ppd.train_ppd_no_fg_dann import create_global_model
from models.experiment_target_learner import FederatedTargetLearner
from utils import test_classification, get_timestamp

if __name__ == "__main__":
    # dann_task_id = '20200118_no_fg_512_0.0006_5'  # DA: 73.5, 35.5; TA: 73.8, 36.7
    # dann_task_id = '20200118_no_fg_512_0.0007_1' # DA: 73.8, 35.6; TA: 74.4, 37.0
    # dann_task_id = '20200118_no_fg_512_0.0008_1'  # DA: 74.6, 38.0; TA: 74.4, 37.6; TA_ft:75.0, 38.5
    # dann_task_id = '20200122_no_fg_pw1.0_bs512_lr0.001_v2'  # DA: 73.99, 37.74; TA: 74.1, 38.9
    # dann_task_id = '20200122_no_fg_pw1.0_bs512_lr0.001_v2'  # DA: 73.97, 37.45; TA: 74.3, 38.4

    # new
    # SOURCE: acc:0.9245521687681071, auc:0.8535690499976215, ks:0.5554809146676617
    # TARGET: acc:0.9401055889313672, auc:0.7306994707337918, ks:0.3536963288042774
    # DA:     acc:0.9401055889313672, auc:0.7244216448218789, ks:0.3333188146132095
    # dann_task_id = '20210509_no_fg_pw1.0_bs64_lr0.0005_v1620511219'

    # SOURCE: acc:0.9245929734361611, auc:0.7939660830420296, ks:0.4574604612255214
    # TARGET: acc:0.9401055889313672, auc:0.7445937387430869, ks:0.3726256006629954
    # DA:     acc:0.9401055889313672, auc:0.7345929647383451, ks:0.37084833274081264
    # dann_task_id = '20210509_no_fg_pw1.0_bs64_lr0.0005_v1620520547'

    # 4000
    # [INFO] SOURCE: acc:0.9245929734361611, auc:0.7648488708891032, ks:0.40441486224618756
    # [INFO] TARGET: acc:0.9401055889313672, auc:0.7405018140552199, ks:0.36271922286392344
    dann_task_id = '20210512_no_fg_s02_tgt4000_pw1.0_bs64_lr0.0005_v1620748815'



    # hyper-parameters
    timestamp = get_timestamp()
    lr = 5e-4
    batch_size = 64
    # batch_size = 128
    # batch_size = 64
    pos_class_weight = 1.0
    metrics = ('ks', 'auc')

    pdd_no_fg_dann_dir = "ppd_no_fg_dann"
    ppd_no_fg_ft_result_dir = "ppd_no_fg_target"

    # Load models
    model = create_global_model(pos_class_weight=pos_class_weight)

    # load pre-trained model
    model.load_model(root=pdd_no_fg_dann_dir,
                     task_id=dann_task_id,
                     load_global_classifier=False,
                     timestamp=None)
    dann_exp_result = None

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    # data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    # target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
    # target_test_file_name = data_dir + 'PPD_2014_10to12_test.csv'
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
    plat_target = FederatedTargetLearner(model=model,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=400)
    plat_target.set_model_save_info(ppd_no_fg_ft_result_dir)

    target_task_id = dann_task_id + "_target_ft_" + str(batch_size) + "_" + str(lr) + "_" + str(timestamp)
    plat_target.train_target_with_alternating(global_epochs=400,
                                              top_epochs=1,
                                              bottom_epochs=1,
                                              lr=lr,
                                              task_id=target_task_id,
                                              dann_exp_result=None,
                                              metric=metrics,
                                              weight_decay=0.00001)

    print("[DEBUG] Global classifier Model Parameter After train:")
    model.print_parameters()

    acc, auc, ks = test_classification(model, target_valid_loader, "test")
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
