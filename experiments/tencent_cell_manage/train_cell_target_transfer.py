from datasets.cell_manage_dataloader import get_cell_manager_dataloader_ob
from models.experiment_target_learner import FederatedTargetLearner
from experiments.tencent_cell_manage.train_cell_dann import create_global_dann_model
# from train_census_target_test import test_classification
from utils import test_classification

if __name__ == "__main__":
    dann_root_folder = "cell_dann/"
    # dann_task_id = "20201231_cell_1024_0.001_v0_1609430897"
    # dann_task_id = "20200103_cell_256_0.001_v0_1609662002"
    # dann_task_id = "20201231_cell_256_0.001_v0_1609383182"
    # dann_task_id = "20200105_cell_256_0.001_v0_1609799043"
    # dann_task_id = "20200105_cell_256_0.001_v1_1609816695"

    # dann_task_id = "20200114_cell_256_0.001_v0_1610565311"
    # dann_task_id = "20200114_cell_256_0.001_v0_1610568602"
    # dann_task_id = "20200114_cell_256_0.001_v0_1610580001"
    # dann_task_id = "20200118_cell_256_0.001_v0_1611012373"  # DA: 69.7, 29.99, TA:

    # dann_task_id = "20200114_cell_512_0.001_v0_1610583874"  # DA: 70.6, 30.3; TA: 69.6, 28.9; TA_ft: 70.7, 30.4
    # dann_task_id = "20200115_cell_256_0.0008_v0_1610656846"  # DA: 70.2, 29.1, TA: 69.56, 28.71

    # dann_task_id = "20200115_cell_512_0.001_v0_1610648731"  # DA: 70.2, 29.8, TA_rt: 70.00, 29.95, TA_ft: 70.2, 30.9
    # dann_task_id = "20200115_cell_256_0.0008_v0_1610678010"  # DA: 70.5, 30.9, TA_rt: 69.79, 29.89, TA_ft: 70.5, 31.2
    # dann_task_id = "20200115_cell_256_0.0008_v0_1610730877"  # DA: 70.2, 29.5, TA_rt: 69.93, 30.04, TA_ft: 70.4, 30.4
    # dann_task_id = "20200121_cell_pw1.0_bs256_lr0.0008_v0_t1611198306"  # DA:69.9,30.3;TA_rt:70.5,30.2;TA_ft:70.4,30.9

    # # DA: 69.58, 30.12 | TA_rt: 69.58, 29.69; 69.94, 30.4;
    # dann_task_id = "20200301_cell_pw1.0_bs512_lr0.0008_v0_t1614577582"

    # DA: 70.0, 30.9
    # dann_task_id = '20200302_cell_pw1.0_bs256_lr0.0008_v0_t1614584956'

    # DA: 70.3, 29.98 | TA_rt:70.35, 30.06 | 70.3, 30.2
    # dann_task_id = '20200302_cell_pw1.0_bs256_lr0.001_v0_t1614584956'

    # DA: 69.84, 29.91 | TA_rt:70.0, 30.08
    # dann_task_id = '20200302_cell_pw1.0_bs512_lr0.001_v0_t1614618166'

    # DA: 70.07, 30.09 | TA_rt:70.17, 30.4
    # dann_task_id = '20200302_cell_pw1.0_bs512_lr0.001_v1_t1614618166'

    # (with att) DA: 69.96, 30.27 | TA_rt:70.22, 30.85 (0.7014, 0.3137)
    dann_task_id = '20210408_cell_pw1.0_bs256_lr0.0008_v0_t1617925826'

    #
    # Load models
    #
    wrapper = create_global_dann_model(pos_class_weight=20.0)

    #
    # load pre-trained model
    #
    load_global_classifier = False
    wrapper.load_model(root=dann_root_folder,
                       task_id=dann_task_id,
                       load_global_classifier=load_global_classifier,
                       timestamp=None)
    # dann_exp_result = load_dann_experiment_result(root=dann_root_folder, task_id=dann_task_id)
    dann_exp_result = None

    print("[DEBUG] Global classifier Model Parameter Before train:")
    wrapper.print_parameters()

    # Load data
    batch_size = 128
    # batch_size = 512
    # target_dir = "../../data/cell_manager/B_train_data_2/"
    # target_dir = "../../data/cell_manager/C_train_data_2/"
    # target_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_3/"
    target_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_4/"

    print("[INFO] Load train data")
    tgt_train_loader = get_cell_manager_dataloader_ob(dir=target_dir, batch_size=batch_size, data_mode="train",
                                                      suffix=None)

    print("[INFO] Load test data")
    tgt_test_loader = get_cell_manager_dataloader_ob(dir=target_dir, batch_size=batch_size * 2, data_mode="test")
    print("[INFO] data loaded ...")

    acc, auc, ks = test_classification(wrapper, tgt_test_loader)
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")

    #
    # perform target training
    #
    plat_target = FederatedTargetLearner(model=wrapper,
                                         target_train_loader=tgt_train_loader,
                                         # target_val_loader=tgt_val_loader,
                                         target_val_loader=tgt_test_loader,
                                         patience=150)
    plat_target.set_model_save_info("cell_manage_target")

    # lr = 2e-4
    lr = 3e-4
    # lr = 5e-4
    # lr = 8e-4
    version = 1
    date = "20200121"
    glr = "ft_glr" if load_global_classifier else "rt_glr"
    tag = "_target_" + date + "_" + glr + "_" + str(batch_size) + "_" + str(lr) + "_v" + str(version)
    target_task_id = dann_task_id + tag
    # plat_target.set_fine_tuning_region_indices(fine_tuning_region_index_list=[1, 2, 5])
    plat_target.train_target_with_alternating(global_epochs=200, top_epochs=1, bottom_epochs=1, lr=lr,
                                              task_id=target_task_id, dann_exp_result=dann_exp_result)
    # plat_target.train_target_as_whole(epochs=200, lr=2e-4, task_id=target_task_id, dann_exp_result=dann_exp_result)

    print("[DEBUG] Global classifier Model Parameter After train:")
    wrapper.print_parameters()

    acc, auc, ks = test_classification(wrapper, tgt_test_loader)
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
