from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census.train_census_no_fg_dann import create_global_model
from experiments.income_census.train_census_target_test import test_classification
from models.experiment_target_learner import FederatedTargetLearner
from utils import get_timestamp

if __name__ == "__main__":

    # DA: acc:0.71777890983189, auc:0.7824169864494814, ks:0.4377377904436571
    # dann_task_id = '20201208_DEGREE_no_fg_dann_0.0005_3'

    # [INFO] SOURCE: acc:0.9359625105307497, auc:0.9288294156813324, ks:0.7148908495012257
    # [INFO] TARGET: acc:0.7371370351502802, auc:0.8009867953323917, ks:0.4568925871571544
    # DA: acc:0.7384105960264901, auc:0.799057470135172, ks:0.47357816933997343
    # dann_task_id = '20210509_DEGREE_no_fg_dann_0.0005_1620504931'

    # [INFO] SOURCE: acc:0.9368365627632688, auc:0.9338279410299093, ks:0.7228686281674197
    # [INFO] TARGET: acc:0.7412124299541518, auc:0.7995789753678901, ks:0.44987330769325784
    # DA:            acc:0.7407030056036679, auc:0.7994547390621064, ks:0.4633270224162511
    # dann_task_id = '20210509_DEGREE_no_fg_dann_0.0005_1620510405'

    # TODO: 4 feature groups
    # SOURCE: acc:0.93356149957877, auc:0.9308896066757771, ks:0.7168777609323624
    # TARGET: acc:0.7320427916454406, auc:0.7960209337457117, ks:0.43479748333150936
    # DA:     acc:0.7412124299541518, auc:0.7943686626916795, ks:0.45968811947597277
    # dann_task_id = '20210511_DEGREE_no_fg_dann_0.0005_1620706796'

    # [INFO] acc:0.9357097725358046, auc:0.9341811794436317, ks:0.7243818010971153
    # [INFO] acc:0.7419765664798778, auc:0.7905918789958488, ks:0.4490879618897592
    # DA:    acc:0.7478349465104432, auc:0.7913430418620336, ks:0.4564576882740173
    # dann_task_id = '20210511_DEGREE_no_fg_dann_0.0005_1620715139'

    # [INFO] SOURCE: acc:0.9405328559393429, auc:0.9340857449454303, ks:0.7241876856500609
    # [INFO] TARGET: acc:0.7338257768721345, auc:0.798348964127736, ks:0.4434279568743467
    # DA:    acc:0.7463066734589914, auc:0.7909110154946372, ks:0.4571350274856669
    # dann_task_id = '20210512_DEGREE_no_fg_dann_0.0005_1620720257'

    # SOURCE: acc:0.9398062342038753, auc:0.9274175264364926, ks:0.7102418612071386
    # TARGET: acc:0.7345899133978604, auc:0.8009753052694291, ks:0.44562026088762463
    # DA:     acc:0.7417218543046358, auc:0.8025083669202238, ks:0.4654047130514764
    dann_task_id = '20210512_DEGREE_no_fg_dann_0.0005_1620757044'

    # Hyper-parameters

    batch_size = 64
    lr = 5e-4
    pos_class_weight = 5.0
    weight_decay = 0.0001

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

    plat_target = FederatedTargetLearner(model=model,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=500)
    plat_target.set_model_save_info("census_target")

    appendix = "_" + str(batch_size) + "_" + str(lr) + "_v" + str(timestamp)
    target_task_id = dann_task_id + "_target_ft" + appendix
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

    acc, auc, ks = test_classification(model, target_valid_loader, 'test')
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
