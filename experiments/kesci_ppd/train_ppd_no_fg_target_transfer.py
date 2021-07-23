from datasets.ppd_dataloader import get_pdd_dataloaders_ob
# from experiments.kesci_ppd.global_config import tgt_data_tag, src_data_tag, tgt_tag
from experiments.kesci_ppd.train_ppd_no_fg_dann import create_global_model
from models.experiment_target_learner import FederatedTargetLearner
from utils import test_classifier, get_timestamp

if __name__ == "__main__":

    # TODO: data_tag = 'lbl004tgt4000v4'
    # SOURCE: acc:0.9201095368496249, auc:0.7239260950825191, ks:0.3349336818883921
    # TARGET: acc:0.8966915342199157, auc:0.7530701315520101, ks:0.40716025698532204
    # DA:     acc:0.8128446318520921, auc:0.7579948497123521, ks:0.4269246871067989
    # dann_task_id = '20210716_no_fg_lbl004tgt4000v4_pw3.0_bs64_lr0.0005_v1626401920'

    # SOURCE: acc:0.9019526134063579, auc:0.7266894106097392, ks:0.3469386891429865
    # TARGET: acc:0.883879338306844, auc:0.7518258333275826, ks:0.40423662339872984
    # DA:     acc:0.8172234836198508, auc:0.7529348473660844, ks:0.41226741471201866
    # dann_task_id = '20210716_no_fg_lbl004tgt4000v4_pw3.0_bs64_lr0.0005_v1626405223'

    # SOURCE: acc:0.9148708179545184, auc:0.7235170779333221, ks:0.33691813424901584
    # TARGET: acc:0.8927992215374635, auc:0.7548480906671451, ks:0.4077341724690788
    # DA:     acc:0.8209536166072008, auc:0.7594425773980811, ks:0.42037526481771575
    # dann_task_id = '20210716_no_fg_lbl004tgt4000v4_pw3.0_bs64_lr0.0005_v1626407351'

    # SOURCE: acc:0.9083819502321705, auc:0.7320917772312796, ks:0.3564598184041445
    # TARGET: acc:0.8889069088550113, auc:0.7589287562710907, ks:0.40614411604364054
    # DA:     acc:0.8181965617904638, auc:0.7602505445871374, ks:0.4153043362247632
    # dann_task_id = '20210716_no_fg_lbl004tgt4000v4_pw3.0_bs64_lr0.0005_v1626412387'

    # SOURCE: acc:0.9102274080247649, auc:0.729957089095709, ks:0.3510151066555607
    # TARGET: acc:0.8897178073305222, auc:0.7534953925742928, ks:0.40745641577704017
    # DA:     acc:0.8365228673370094, auc:0.7575178327847207, ks:0.4207697598295966
    dann_task_id = '20210717_no_fg_lbl004tgt4000v4_pw3.0_bs64_lr0.0005_v1626453974'

    # TODO: data_tag = 'lbl002tgt4000'
    # SOURCE: acc:0.9199904750565544, auc:0.7280825426180655, ks:0.34471230781143913
    # TARGET: acc:0.8983133311709374, auc:0.7535720200674898, ks:0.4049462543676234
    # DA:     acc:0.8292799221537464, auc:0.7557966613990657, ks:0.4108884073672806
    # dann_task_id = '20210716_no_fg_lbl002tgt4000_pw3.0_bs64_lr0.0005_v1626399952'

    # SOURCE: acc:0.9153470651268009, auc:0.7280986896891153, ks:0.34839726591041953
    # TARGET: acc:0.892312682452157, auc:0.7566077914490043, ks:0.4063241115811314
    # DA:     acc:0.832685695750892, auc:0.7575741892149894, ks:0.4138931252055859
    # dann_task_id = '20210716_no_fg_lbl002tgt4000_pw3.0_bs64_lr0.0005_v1626408165'

    # SOURCE: acc:0.8862364567210382, auc:0.7281803319001485, ks:0.34107113069449385
    # TARGET: acc:0.8696075251378528, auc:0.7502744500647525, ks:0.39781946620109354
    # DA:     acc:0.8163023029516705, auc:0.7525274133778663, ks:0.4090982281078271
    # dann_task_id = '20210716_no_fg_lbl002tgt4000_pw3.0_bs64_lr0.0005_v1626408821'

    # SOURCE: acc:0.9134420764376712, auc:0.7258754319719363, ks:0.34265822907892524
    # TARGET: acc:0.8913396042815439, auc:0.7466922225825966, ks:0.3958188129265549
    # DA:     acc:0.8190074602659747, auc:0.7567538581152109, ks:0.4133818918738629
    # dann_task_id = '20210716_no_fg_lbl002tgt4000_pw3.0_bs64_lr0.0005_v1626410479'

    # SOURCE: acc:0.9128467674723182, auc:0.7251562449540402, ks:0.3406561937922598
    # TARGET: acc:0.892312682452157, auc:0.7502392272958345, ks:0.40143145332882485
    # DA:     acc:0.8203048978267921, auc:0.7533580956587147, ks:0.41508983675037325
    # dann_task_id = '20210716_no_fg_lbl002tgt4000_pw3.0_bs64_lr0.0005_v1626411398'

    # TODO: data_tag = 'lbl001tgt4000'
    # SOURCE: acc:0.9145731634718419, auc:0.7293204763350694, ks:0.3454514456565916
    # TARGET: acc:0.8949075575737918, auc:0.7560264001122527, ks:0.39466753156535167
    # DA:     acc:0.8170613039247486, auc:0.7565034170398843, ks:0.4014809089717137
    # dann_task_id = '20210716_no_fg_lbl001tgt4000_pw3.0_bs64_lr0.0005_v1626415163'

    # SOURCE: acc:0.9129658292653887, auc:0.7282523765541304, ks:0.354071512934207
    # TARGET: acc:0.8910152448913397, auc:0.7500201272965246, ks:0.39660492762224175
    # DA:     acc:0.8141420694129095, auc:0.7499105772968696, ks:0.4066680008188935
    # dann_task_id = '20210716_no_fg_lbl001tgt4000_pw3.0_bs64_lr0.0005_v1626416325'

    # SOURCE: acc:0.9111799023693297, auc:0.7250392605578544, ks:0.3343657886438581
    # TARGET: acc:0.88582549464807, auc:0.751229202752034, ks:0.4101488729864078
    # DA:     acc:0.8344145313006811, auc:0.7529408855550419, ks:0.41566432729974495
    # dann_task_id = '20210716_no_fg_lbl001tgt4000_pw3.0_bs64_lr0.0005_v1626419039'

    # SOURCE: acc:0.9185022026431718, auc:0.7281900050285153, ks:0.3428477619378621
    # TARGET: acc:0.892312682452157, auc:0.753153084766972, ks:0.40719476092222123
    # DA:     acc:0.8037625689263704, auc:0.7535961728233191, ks:0.4107900711471179
    # dann_task_id = '20210717_no_fg_lbl001tgt4000_pw3.0_bs64_lr0.0005_v1626424437'

    # SOURCE: acc:0.918740326229313, auc:0.719178805813006, ks:0.34567543903533515
    # TARGET: acc:0.8952319169639961, auc:0.7500307660104018, ks:0.3956382423234491
    # DA:     acc:0.8032760298410639, auc:0.7508316886456743, ks:0.4064241729981391
    # dann_task_id = '20210717_no_fg_lbl001tgt4000_pw3.0_bs64_lr0.0005_v1626452297'

    # hyper-parameters
    timestamp = get_timestamp()
    lr = 8e-4
    batch_size = 64
    # batch_size = 128
    metrics = ('ks', 'auc')
    pos_class_weight = 5.0

    weight_decay = 0.0

    pdd_no_fg_dann_dir = "ppd_no_fg_dann"
    ppd_no_fg_ft_result_dir = "ppd_no_fg_target"

    # initialize model
    model = create_global_model(aggregation_dim=5,
                                num_wide_feature=6,
                                pos_class_weight=pos_class_weight)

    # load pre-trained model
    model.load_model(root=pdd_no_fg_dann_dir,
                     task_id=dann_task_id,
                     load_global_classifier=False,
                     timestamp=None)
    dann_exp_result = None

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    ts = '20210522'
    data_tag = 'lbl004tgt4000v4'
    # data_tag = 'lbl002tgt4000'
    # data_tag = 'lbl001tgt4000'
    data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{ts}/"
    source_train_file_name = data_dir + f"PPD_2014_src_1to9_da_{data_tag}_train.csv"
    source_test_file_name = data_dir + f'PPD_2014_src_1to9_da_{data_tag}_test.csv'
    target_train_file_name = data_dir + f'PPD_2014_tgt_10to12_da_{data_tag}_train.csv'
    target_test_file_name = data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_test.csv'

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
                                              weight_decay=weight_decay)

    print("[DEBUG] Global classifier Model Parameter After train:")
    model.print_parameters()

    acc, auc, ks = test_classifier(model, target_valid_loader, "test")
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
