from datasets.cell_manage_dataloader import get_cell_manager_dataloader_ob
from models.experiment_target_learner import FederatedTargetLearner
from experiments.tencent_cell_manage.train_cell_dann_no_fg import create_global_daan_model
from utils import test_classifier

if __name__ == "__main__":
    dann_root_folder = "cell_dann/"
    # dann_task_id = "20200106_cell_512_0.0003_v0_1609870117"
    # dann_task_id = "20200106_cell_no_fg_256_0.0003_v0_1609874107"
    # dann_task_id = "20200106_cell_no_fg_256_0.0003_v0_1609877854"
    # dann_task_id = "20200106_cell_no_fg_128_0.0003_v0_1609877854"
    # dann_task_id = "20200118_cell_no_fg_128_0.0003_v0_1610999616"  # DA: 70.3, 30.6; TA: 69.3, 29.3
    dann_task_id = "20200118_cell_no_fg_256_0.0003_v0_1610999616"  # DA: 69.7, 28.4; TA: 68.7, 26.8
    # dann_task_id = "20200118_cell_no_fg_128_0.0003_v0_1611008578"  # DA: 69.9, 28.6; TA: 70.1, 29.5
    #
    # Load models
    wrapper = create_global_daan_model(pos_class_weight=20.0)

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
    # batch_size = 128
    batch_size = 512
    # target_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_3/"
    target_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_4/"

    print("[INFO] Load train data")
    # tgt_train_loader = get_cell_manager_dataloader_ob(dir=target_dir, batch_size=batch_size, data_mode="train_bal",
    #                                                   suffix="_1500_4795")
    tgt_train_loader = get_cell_manager_dataloader_ob(dir=target_dir, batch_size=batch_size, data_mode="train",
                                                      suffix=None)
    # print("[INFO] Load val data")
    # tgt_val_loader = get_cell_manager_dataloader(dir=target_dir, batch_size=batch_size * 2, data_mode="val")
    print("[INFO] Load test data")
    tgt_test_loader = get_cell_manager_dataloader_ob(dir=target_dir, batch_size=batch_size * 2, data_mode="test")
    print("[INFO] data loaded ...")

    acc, auc, ks = test_classifier(wrapper, tgt_test_loader)
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")

    #
    # perform target training
    #
    plat_target = FederatedTargetLearner(model=wrapper,
                                         target_train_loader=tgt_train_loader,
                                         # target_val_loader=tgt_val_loader,
                                         target_val_loader=tgt_test_loader,
                                         patience=50)
    plat_target.set_model_save_info("cell_manage_target")
    lr = 2e-4
    version = 2
    glr = "ft_glr" if load_global_classifier else "rt_glr"
    tag = "_target_" + glr + "_" + str(batch_size) + "_" + str(lr) + "_v" + str(version)
    target_task_id = dann_task_id + tag
    # plat_target.set_fine_tuning_region_indices(fine_tuning_region_index_list=[1, 2, 5])
    plat_target.train_target_with_alternating(global_epochs=200, top_epochs=1, bottom_epochs=1, lr=lr,
                                              task_id=target_task_id, dann_exp_result=dann_exp_result)
    # plat_target.train_target_as_whole(epochs=200, lr=2e-4, task_id=target_task_id, dann_exp_result=dann_exp_result)

    print("[DEBUG] Global classifier Model Parameter After train:")
    wrapper.print_parameters()

    acc, auc, ks = test_classifier(wrapper, tgt_test_loader)
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
