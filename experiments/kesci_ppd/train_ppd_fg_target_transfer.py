from datasets.ppd_dataloader import get_pdd_dataloaders_ob
# from experiments.kesci_ppd.global_config import tgt_data_tag, tgt_tag, data_tag
from experiments.kesci_ppd.train_ppd_fg_dann import create_pdd_global_model
from models.experiment_target_learner import FederatedTargetLearner
from utils import test_classifier, get_timestamp, get_current_date

if __name__ == "__main__":
    dann_root_folder = "ppd_dann"

    # TODO: lbl004tgt4000v4
    # data_tag = 'lbl004tgt4000v4'
    # SOURCE: acc:0.9091558518871294, auc:0.7459006214027738, ks:0.36889452415790763
    # TARGET: acc:0.8908530651962374, auc:0.7668830638575863, ks:0.4197357918538505
    # DA:     acc:0.9004216672072657, auc:0.7650488920785862, ks:0.4250373217584127
    # dann_task_id = '20210716_PPD_fg_dann_src_t004_tgt3000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626383450_ve0'

    # data_tag = 'lbl004tgt4000v4'
    # SOURCE: acc:0.9132634837480652, auc:0.7451016562662902, ks:0.3773429941687662
    # TARGET: acc:0.8953940966590983, auc:0.7633224013359925, ks:0.4274330451104471
    # DA:     acc:0.898799870256244, auc:0.7641003213466656, ks:0.42564516611345354
    # dann_task_id = '20210716_PPD_fg_dann_src_t004_tgt3000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626385714_ve0'

    # data_tag = 'lbl004tgt4000v4'
    # SOURCE: acc:0.9153470651268009, auc:0.7430744809220465, ks:0.373563974972594
    # TARGET: acc:0.8926370418423614, auc:0.7650928845981326, ks:0.4289788214835313
    # DA:     acc:0.898799870256244, auc:0.7631264477276857, ks:0.4250545737268623
    # dann_task_id = '20210716_PPD_fg_dann_src_lbl004tgt4000v4_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626400861_ve0'

    # data_tag = 'lbl004tgt4000v4'
    # SOURCE: acc:0.9068936778187879, auc:0.7467287066861117, ks:0.3677212039155413
    # TARGET: acc:0.888582549464807, auc:0.7622677309981069, ks:0.4152525803194144
    # DA:     acc:0.8986376905611417, auc:0.7642672605944338, ks:0.42530012674446155
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl004tgt4000v4_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626452461_ve0'

    # data_tag = 'lbl004tgt4000v4'
    # SOURCE: acc:0.9018335516132873, auc:0.7412806446092135, ks:0.36437495645202955
    # TARGET: acc:0.8824197210509245, auc:0.765942256511468, ks:0.42454736585444397
    # DA:     acc:0.8991242296464482, auc:0.7647846494284998, ks:0.42927440520963445
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl004tgt4000v4_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626467345_ve0'

    # data_tag = 'lbl004tgt4000v4'
    # SOURCE: acc:0.9079652339564234, auc:0.7400313248140278, ks:0.3622347264199989
    # TARGET: acc:0.887609471294194, auc:0.7594506283166909, ks:0.42725419970418627
    # DA:     acc:0.8994485890366526, auc:0.7609906540336252, ks:0.4287476451063066
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl004tgt4000v4_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626485700_ve0'

    # use_interaction = True
    # SOURCE: acc:0.9196928205738778, auc:0.7406516393609872, ks:0.3617964127908784
    # TARGET: acc:0.8983133311709374, auc:0.7625211911679122, ks:0.4212781178332448
    # DA:     acc:0.898799870256244, auc:0.769052722015582, ks:0.4406169993996315
    # dann_task_id = '20210716_PPD_fg_dann_src_lbl004tgt4000v4_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626415417_ve0'

    # use_interaction = True
    # SOURCE: acc:0.9001666865102989, auc:0.7390654226419016, ks:0.3560408510317575
    # TARGET: acc:0.8820953616607201, auc:0.7661085942406027, ks:0.43139122173840033
    # DA:     acc:0.898799870256244, auc:0.7651879797484893, ks:0.43301578210073766
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl004tgt4000v4_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep800_ts1626493717_ve0'

    # use_interaction = True
    # SOURCE: acc:0.8986784140969163, auc:0.7391552769358714, ks:0.3597230885633848
    # TARGET: acc:0.8807979240999027, auc:0.7638720202975159, ks:0.42402003068550126
    # DA:     acc:0.9000973078170613, auc:0.7649206788994626, ks:0.42393894643378804
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl004tgt4000v4_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep800_ts1626501076_ve0'

    # use_interaction = True
    # SOURCE: acc:0.8896892487200857, auc:0.7346335176618477, ks:0.36217784640996703
    # TARGET: acc:0.8657152124554006, auc:0.760609960596504, ks:0.4310950629466822
    # DA:     acc:0.8981511514758352, auc:0.7656388702951007, ks:0.4372344634522799
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl004tgt4000v4_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep800_ts1626498133_ve0'

    # use_interaction = True
    # SOURCE: acc:0.904929158233123, auc:0.7445431590553, ks:0.36944684971147623
    # TARGET: acc:0.8879338306843984, auc:0.7653864555945832, ks:0.4321784865653171
    # DA:     acc:0.8944210184884852, auc:0.7643081293575598, ks:0.434788107413056
    # dann_task_id = '20210718_PPD_fg_dann_src_lbl004tgt4000v4_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626546545_ve0'

    # TODO: lbl002tgt4000
    # data_tag = 'lbl002tgt4000'
    # SOURCE: acc:0.9174901774020717, auc:0.7404992623987716, ks:0.36049527626379047
    # TARGET: acc:0.8947453778786896, auc:0.7616988473384814, ks:0.41917682807608353
    # DA:     acc:0.9000973078170613, auc:0.7595458016759712, ks:0.41817448870916174
    # dann_task_id = '20210716_PPD_fg_dann_src_lbl002tgt4000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626405321_ve0'

    # data_tag = 'lbl002tgt4000'
    # SOURCE: acc:0.9021907369924991, auc:0.7486606369825564, ks:0.3879847956559189
    # TARGET: acc:0.8816088225754135, auc:0.7602735472117369, ks:0.42203432911695227
    # DA:     acc:0.8981511514758352, auc:0.7579272795025914, ks:0.4247078091610253
    # dann_task_id = '20210716_PPD_fg_dann_src_lbl002tgt4000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626419335_ve0'

    # SOURCE: acc:0.9129658292653887, auc:0.7358059562388681, ks:0.3529561307002876
    # TARGET: acc:0.8926370418423614, auc:0.7599946403884683, ks:0.42945382568151025
    # DA:     acc:0.8981511514758352, auc:0.7577039976261293, ks:0.4105594698355083
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl002tgt4000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626456960_ve0'

    # SOURCE: acc:0.911060840576259, auc:0.731048867888796, ks:0.3492363594153622
    # TARGET: acc:0.8863120337333766, auc:0.757161867169044, ks:0.43066836426036215
    # DA:     acc:0.8981511514758352, auc:0.7593826406092936, ks:0.427910924636501
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl002tgt4000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626459408_ve0'

    # SOURCE: acc:0.9026074532682462, auc:0.7315533316092997, ks:0.35371970327074165
    # TARGET: acc:0.8775543301978592, auc:0.7614400678117372, ks:0.4282226101998238
    # DA:     acc:0.8981511514758352, auc:0.7575860827220386, ks:0.4236272608704653
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl002tgt4000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626462983_ve0'

    # use_interaction = True
    # SOURCE: acc:0.8830813192046673, auc:0.7449259277676269, ks:0.3668501185537605
    # TARGET: acc:0.8653908530651963, auc:0.7617112112492035, ks:0.4164521671922767
    # DA:     acc:0.8981511514758352, auc:0.7598802023310859, ks:0.4224926564120966
    # dann_task_id = '20210718_PPD_fg_dann_src_lbl002tgt4000_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626548731_ve0'

    # use_interaction = True
    # SOURCE: acc:0.8988570067865223, auc:0.740690810492785, ks:0.361519418729625
    # TARGET: acc:0.8798248459292897, auc:0.7636733351275381, ks:0.4270253235894215
    # DA:     acc:0.8981511514758352, auc:0.7590871868480195, ks:0.42290382832681206
    # dann_task_id = '20210718_PPD_fg_dann_src_lbl002tgt4000_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626557796_ve0'

    # use_interaction = True
    # SOURCE: acc:0.9131444219549947, auc:0.74223904003068, ks:0.3636230722450188
    # TARGET: acc:0.8921505027570548, auc:0.7659940124168167, ks:0.4233656060156464
    # DA:     acc:0.8981511514758352, auc:0.7603667078413645, ks:0.4252060000046005
    # dann_task_id = '20210718_PPD_fg_dann_src_lbl002tgt4000_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626560639_ve0'

    # use_interaction = True
    # SOURCE: acc:0.8957614001666865, auc:0.7422659686093885, ks:0.3674986212013512
    # TARGET: acc:0.8744729159909179, auc:0.7665768414176057, ks:0.43538102697517783
    # DA:     acc:0.8983133311709374, auc:0.7600993023303957, ks:0.42636802359149184
    # dann_task_id = '20210718_PPD_fg_dann_src_lbl002tgt4000_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626565828_ve0'

    # use_interaction = True
    # SOURCE: acc:0.9114180259554708, auc:0.7464377319315156, ks:0.3708100554789178
    # TARGET: acc:0.8906908855011353, auc:0.7648596954912555, ks:0.42429778737753976
    # DA:     acc:0.8983133311709374, auc:0.7601415696530975, ks:0.42714109579903064
    # dann_task_id = '20210718_PPD_fg_dann_src_lbl002tgt4000_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626567411_ve0'

    # TODO: lbl001tgt4000v4
    # SOURCE: acc:0.9117156804381474, auc:0.7392268933523994, ks:0.3561727985483859
    # TARGET: acc:0.8934479403178722, auc:0.7605915584968244, ks:0.4097210241688577
    # DA:     acc:0.8981511514758352, auc:0.7559506478689219, ks:0.4069129787708778
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl001tgt4000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626469155_ve0'

    # SOURCE: acc:0.8951065603047982, auc:0.7402194973892882, ks:0.36645346990984495
    # TARGET: acc:0.8726889393447941, auc:0.7592390041703758, ks:0.41553148714268295
    # DA:     acc:0.8981511514758352, auc:0.7568790786528744, ks:0.41211904778335207
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl001tgt4000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626478167_ve0'

    # SOURCE: acc:0.9160019049886892, auc:0.7346506723504357, ks:0.35021707356529996
    # TARGET: acc:0.8952319169639961, auc:0.7624544835565737, ks:0.4294544007471252
    # DA:     acc:0.8981511514758352, auc:0.7579340365235674, ks:0.42200212544251303
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl001tgt4000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626479874_ve0'

    # SOURCE: acc:0.9117156804381474, auc:0.7411405857714024, ks:0.3633315684912151
    # TARGET: acc:0.8927992215374635, auc:0.7636724725291156, ks:0.4196219288620832
    # DA:     acc:0.8981511514758352, auc:0.755128239632142, ks:0.4079129787708778
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl001tgt4000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626480963_ve0'

    # SOURCE: acc:0.9046315037504464, auc:0.7400473459328852, ks:0.36758855106663624
    # TARGET: acc:0.8850145961725592, auc:0.7648223162262814, ks:0.4347076251400285
    # DA:     acc:0.8981511514758352, auc:0.7557906232101083, ks:0.42188596218828567
    # dann_task_id = '20210717_PPD_fg_dann_src_lbl001tgt4000_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626483036_ve0'

    # use_interaction = True
    # [INFO] SOURCE: acc:0.9176092391951423, auc:0.7322909328377064, ks:0.3505288808124985
    # [INFO] TARGET: acc:0.8918261433668505, auc:0.7593105998394417, ks:0.4234311634957549
    # DA : acc:0.8981511514758352, auc:0.7566722908083813, ks:0.4096571918855942
    # dann_task_id = '20210718_PPD_fg_dann_src_lbl001tgt4000_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626578103_ve0'

    # use_interaction = True
    # SOURCE: acc:0.9113584950589356, auc:0.7462904434378681, ks:0.3738168869746843
    # TARGET: acc:0.8887447291599092, auc:0.7656207948326904, ks:0.41953854434791005
    # DA: acc:0.8981511514758352, auc:0.7577783375083097, ks:0.41468096509811764
    # dann_task_id = '20210718_PPD_fg_dann_src_lbl001tgt4000_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626584858_ve0'

    # use_interaction = True
    # SOURCE: acc:0.9079652339564234, auc:0.7434938513414489, ks:0.3674572081205308
    # TARGET: acc:0.8877716509892961, auc:0.7634326701676661, ks:0.4220682579882365
    # DA:     acc:0.8981511514758352, auc:0.7574688348940614, ks:0.41567985407134955
    # dann_task_id = '20210718_PPD_fg_dann_src_lbl001tgt4000_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626587448_ve0'

    # use_interaction = True
    # SOURCE: acc:0.9094535063698059, auc:0.7427958242918551, ks:0.3698345305593018
    # TARGET: acc:0.8892312682452157, auc:0.7657473092679874, ks:0.4312261779068992
    # DA:     acc:0.8981511514758352, auc:0.7566867192046612, ks:0.41556790029742396
    # dann_task_id = '20210718_PPD_fg_dann_src_lbl001tgt4000_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626590894_ve0'

    # use_interaction = True
    # SOURCE: acc:0.9002262174068342, auc:0.7405684605332079, ks:0.36240325045326416
    # TARGET: acc:0.8816088225754135, auc:0.7650316401101366, ks:0.42897767135230125
    # DA:     acc:0.8981511514758352, auc:0.756345849061378, ks:0.41512146535919753
    dann_task_id = '20210718_PPD_fg_dann_src_lbl001tgt4000_intrTrue_pw3.0_bs64_lr0.0005_gdFalse_mep600_ts1626593992_ve0'

    # hyper-parameters
    timestamp = get_timestamp()
    # lr = 8e-4
    lr = 6e-4
    # lr = 3e-4
    batch_size = 64
    metrics = ('ks', 'auc')
    load_global_classifier = False
    pos_class_weight = 5.0
    # pos_class_weight = 3.0

    # weight_decay = 0.00001
    weight_decay = 0.0

    # Initialize models
    using_interaction = True
    model = create_pdd_global_model(pos_class_weight=pos_class_weight,
                                    using_interaction=using_interaction)

    # load pre-trained model,
    model.load_model(root=dann_root_folder,
                     task_id=dann_task_id,
                     load_global_classifier=load_global_classifier,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    # ts = '1620085151'
    # data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{ts}/"
    # target_train_file_name = data_dir + f'PPD_2014_tgt_10to11_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_train.csv'
    # target_test_file_name = data_dir + f'PPD_2014_tgt_10to11_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_test.csv'

    ts = '20210522'
    # data_tag = 'lbl004tgt4000v4'
    # data_tag = 'lbl002tgt4000'
    data_tag = 'lbl001tgt4000'
    data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{ts}/"
    target_train_file_name = data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_train.csv'
    target_test_file_name = data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_test.csv'

    print(f"load target train from: {target_train_file_name}.")
    print(f"load target test from: {target_test_file_name}.")

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
                                         patience=1200,
                                         max_global_epochs=500)
    plat_target.set_model_save_info(plat_target_dir)

    date = get_current_date() + "_PPD"
    glr = "ft_glr" if load_global_classifier else "rt_glr"
    using_intr_tag = "intr" + str(True) if using_interaction else str(False)
    tag = "_target_" + date + "_" + using_intr_tag + "_" + glr + "_pw" + str(pos_class_weight) + "_" + str(batch_size) \
          + "_" + str(lr) + "_v" + str(timestamp)
    target_task_id = dann_task_id + tag
    plat_target.train_target_with_alternating(global_epochs=400,
                                              top_epochs=1,
                                              bottom_epochs=1,
                                              lr=lr,
                                              task_id=target_task_id,
                                              dann_exp_result=None,
                                              metric=metrics,
                                              weight_decay=weight_decay)
    # plat_target.train_target_as_whole(global_epochs=200, lr=1e-3, task_id=target_task_id,
    #                                   dann_exp_result=dann_exp_result)

    # load best model
    model.load_model(root=plat_target_dir,
                     task_id=target_task_id,
                     load_global_classifier=True,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter After train:")
    model.print_parameters()

    acc, auc, ks = test_classifier(model, target_valid_loader, "test")
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
