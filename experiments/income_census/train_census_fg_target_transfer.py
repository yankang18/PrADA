from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census.train_census_fg_dann import create_global_model
from experiments.income_census.train_census_target_test import test_classifier
from models.experiment_target_learner import FederatedTargetLearner
from utils import get_timestamp, create_id_from_hyperparameters

if __name__ == "__main__":

    # TODO all4000pos001
    # [INFO] SOURCE: acc:0.9261768082663605, auc:0.9309067105228662, ks:0.7124888378619723
    # [INFO] TARGET: acc:0.6906326504316455, auc:0.7781468384852483, ks:0.43334979028161197
    # DA:  acc:0.6370313103981445, auc:0.7844507041161326, ks:0.4294985628172917
    # FT:  acc:0.6487566035304729, auc:0.7956764322714724, ks:0.4453588390183043
    # dann_task_id = '20210628_src_lr0.0006_bs128_pw1.0_me600_ts1624855126'

    # [INFO] SOURCE: acc:0.9243628013777268, auc:0.9265208474393318, ks:0.700318918229366
    # [INFO] TARGET: acc:0.6853498260533436, auc:0.7767625736254571, ks:0.4341825463293251
    # DA:  acc:0.6370313103981445, auc:0.7911162348799992, ks:0.45195411182453277
    # dann_task_id = '20210628_src_lr0.0006_bs128_pw1.0_me600_ts1624863789'

    # [INFO] SOURCE: acc:0.9240642939150402, auc:0.9280464995227578, ks:0.7060849598163031
    # [INFO] TARGET: acc:0.6861229223038269, auc:0.7809965759208136, ks:0.43767103163525195
    # DA： acc:0.6370313103981445, auc:0.7879926098824406, ks:0.4344065676147936
    # dann_task_id = '20210629_src_lr0.0006_bs128_pw1.0_me600_ts1624910369'

    # [INFO] SOURCE: acc:0.9257864523536166, auc:0.9290178446283113, ks:0.7130501339456564
    # [INFO] TARGET: acc:0.6879268135549542, auc:0.78163708293268, ks:0.4330602858511603
    # DA: acc:0.6370313103981445, auc:0.7870490638207921, ks:0.43449380667307713
    # dann_task_id = '20210629_src_lr0.0006_bs128_pw1.0_me600_ts1624912981'

    # [INFO] SOURCE: acc:0.924638346727899, auc:0.9276720814661421, ks:0.7029978313560403
    # [INFO] TARGET: acc:0.6759438216724649, auc:0.7829266054571584, ks:0.43742417741107215
    # DA: acc:0.6462797039345539, auc:0.7822647823553668, ks:0.4327873869754415
    # dann_task_id = '20210713_src_lr0.0006_bs128_pw1.0_me600_ts1626079670'

    # all4000pos001,using_interaction = True
    # [INFO] SOURCE: acc:0.9255797933409874, auc:0.9270989337805658, ks:0.7063656078581451
    # [INFO] TARGET: acc:0.682257441051411, auc:0.8000822560207156, ks:0.4614028557544176
    # dann_task_id = '20210715_src_intrTrue_lr0.0005_bs128_pw1.0_me800_ts1626294666'

    # all4000pos001,using_interaction = True
    # SOURCE: acc:0.9247301951779564, auc:0.9284769095146599, ks:0.705064421482332
    # TARGET: acc:0.6870248679293905, auc:0.7803901388127791, ks:0.4414108946720845
    # DA:     acc:0.6370313103981445, auc:0.7946908104171047, ks:0.45696163377000254
    # dann_task_id = '20210715_src_intrTrue_lr0.0005_bs128_pw1.0_me800_ts1626299682'

    # all4000pos001,using_interaction = True
    # SOURCE: acc:0.9250516647531573, auc:0.9300641424746501, ks:0.7106773823191734
    # TARGET: acc:0.6778765622986729, auc:0.8001431438572788, ks:0.45846709988936796
    # DA:     acc:0.6370313103981445, auc:0.7889003987004468, ks:0.44345164242067064
    # dann_task_id = '20210715_src_intrTrue_lr0.0005_bs128_pw1.0_me600_ts1626302184'

    # all4000pos001,using_interaction = True
    # SOURCE: acc:0.9251435132032148, auc:0.9297150943448731, ks:0.7081515499425948
    # TARGET: acc:0.6768457672980286, auc:0.8011284067031765, ks:0.4632732180829982
    # DA:     acc:0.6370313103981445, auc:0.787229465577119, ks:0.4418457975330086
    # dann_task_id = '20210715_src_intrTrue_lr0.0005_bs128_pw1.0_me600_ts1626302184'

    # all4000pos001,using_interaction = True
    # SOURCE: acc:0.923926521239954, auc:0.9270415182470292, ks:0.7057787983161118
    # TARGET: acc:0.680324700425203, auc:0.7872031143553989, ks:0.4427616999424438
    # DA:     acc:0.6370313103981445, auc:0.7907684274739705, ks:0.4488132903212464
    # dann_task_id = '20210715_src_intrTrue_lr0.0005_bs128_pw1.0_me600_ts1626307654'

    # all4000pos001,using_interaction = True
    # SOURCE: acc:0.9246613088404133, auc:0.9293590717553567, ks:0.7070289577752263
    # TARGET: acc:0.6774900141734312, auc:0.8005963561501885, ks:0.45592079641290223
    # DA:     acc:0.6370313103981445, auc:0.7832600884252223, ks:0.4330775182577348
    dann_task_id = '20210715_src_intrTrue_lr0.0005_bs128_pw1.0_me600_ts1626309490'

    # TODO all4000pos002
    # SOURCE: acc:0.9241561423650976, auc:0.9266042949021854, ks:0.7013139431049878
    # TARGET: acc:0.6956910880961514, auc:0.7928757996685623, ks:0.44460263679162815
    # DA: acc:0.6373865234624728, auc:0.8069219104235421, ks:0.47488218909604846
    # FT: acc:0.6550313259174019, auc:0.8053878155603749, ks:0.4635813100334889
    # dann_task_id = '20210629_src_lr0.0006_bs128_pw1.0_me600_ts1624917100'

    # SOURCE: acc:0.9247531572904707, auc:0.9295030924449846, ks:0.7082025768592933
    # TARGET: acc:0.6877637130801688, auc:0.7882639674310632, ks:0.4547972617062651
    # DA:     acc:0.638792993223373, auc:0.8040125652705506, ks:0.47280058966274024
    # FT:     acc:0.6404551847589822, auc:0.8048895697815668, ks:0.4668251057596505
    # dann_task_id = '20210629_src_lr0.0006_bs128_pw1.0_me600_ts1624919877'

    # SOURCE: acc:0.9221125143513204, auc:0.920445088285793, ks:0.6921290980992474
    # TARGET: acc:0.6846950517836594, auc:0.780937420613075, ks:0.45319714662528127
    # DA:     acc:0.6372586625751183, auc:0.7960858250101417, ks:0.4654585190763421
    # FT:     acc:0.6372586625751183, auc:0.7968718086102529, ks:0.4657837756181077
    # dann_task_id = '20210629_src_lr0.0006_bs128_pw1.0_me600_ts1624932395'

    # SOURCE: acc:0.9243398392652123, auc:0.9287579793633909, ks:0.7065442020665901
    # TARGET: acc:0.6884030175169416, auc:0.7889693264480883, ks:0.4582351929416997
    # DA:     acc:0.6380258278992457, auc:0.8087794937455126, ks:0.4808914080220612
    # dann_task_id = '20210713_src_lr0.0006_bs128_pw1.0_me600_ts1626112390'

    # SOURCE: acc:0.9239035591274397, auc:0.9281371645701546, ks:0.7052940426074754
    # TARGET: acc:0.6956910880961514, auc:0.7978330799552577, ks:0.4657617806660552
    # DA:     acc:0.6373865234624728, auc:0.7989431885240382, ks:0.4742595410000051
    # dann_task_id = '20210713_src_False_lr0.0006_bs128_pw1.0_me600_ts1626161175'

    # all4000pos002, using_interaction = True
    # SOURCE: acc:0.9262686567164179, auc:0.9315318781288587, ks:0.7154994259471872
    # TARGET: acc:0.6922388441375783, auc:0.7944784254273527, ks:0.4609844912249335
    # DA:     acc:0.6428845416187189, auc:0.8141057729464635, ks:0.4878106238871686
    # dann_task_id = '20210713_src_lr0.0006_bs128_pw1.0_me600_ts1626117256'

    # all4000pos002, using_interaction = True
    # SOURCE: acc:0.9248220436280138, auc:0.9283602271840187, ks:0.7045796657736957
    # TARGET: acc:0.6806035033883135, auc:0.7931788490883198, ks:0.4649812073998091
    # DA:     acc:0.6381536887866002, auc:0.8067643035082727, ks:0.4731099334578441
    # dann_task_id = '20210713_src_intrTrue_lr0.0006_bs128_pw1.0_me600_ts1626120169'

    # all4000pos002, using_interaction = True
    # SOURCE: acc:0.9246842709529277, auc:0.9285871417149585, ks:0.7017221584385764
    # TARGET: acc:0.6748497634573584, auc:0.8152397152735776, ks:0.485012950854083
    # DA:     acc:0.637897967011891, auc:0.8096278906741969, ks:0.4798125945217152
    # dann_task_id = '20210713_src_intrTrue_lr0.0006_bs128_pw1.0_me600_ts1626124918'

    # all4000pos002, using_interaction = True
    # SOURCE: acc:0.9244316877152698, auc:0.931508057756494, ks:0.713101160862355
    # TARGET: acc:0.6863572433192686, auc:0.7991225782214048, ks:0.4710760722645211
    # DA: acc:0.6395601585475004, auc:0.8099916207012245, ks:0.47339707013093996
    # dann_task_id = '20210713_src_intrTrue_lr0.0006_bs128_pw1.0_me600_ts1626126702'

    # all4000pos002, using_interaction = True
    # SOURCE: acc:0.9251894374282434, auc:0.9314552030099437, ks:0.7156014797805843
    # TARGET: acc:0.6900652090525509, auc:0.7992481828350545, ks:0.4630288194693941
    # DA:     acc:0.6393044367727913, auc:0.8137531818420991, ks:0.4877616126274505
    # dann_task_id = '20210713_src_intrTrue_lr0.0006_bs128_pw1.0_me600_ts1626137678'

    # TODO all4000pos004v4
    # all4000pos004v4, # param = 20235
    # SOURCE: acc:0.924638346727899, auc:0.9282906349057796, ks:0.7034060466896288
    # TARGET: acc:0.6880924555252564, auc:0.7895072339216861, ks:0.44667884373900957
    # DA:     acc:0.6586157641864693, auc:0.8121414898624195, ks:0.4830808002748226
    # FT:     acc:0.6595247370471368, auc:0.8130160213022459, ks:0.48531402498948717
    # dann_task_id = '20210704_src_lr0.0006_bs128_pw1.0_me600_ts1625369952'

    # all4000pos004v4, # param = 20235
    # SOURCE: acc:0.9248679678530425, auc:0.9279422399908375, ks:0.7067738231917337
    # TARGET: acc:0.6849759771458253, auc:0.7874327962084755, ks:0.4567111493845947
    # DA:     acc:0.6584859109206597, auc:0.8151877506837027, ks:0.4951468626663035
    # FT:     acc:0.6591351772497078, auc:0.8148384630234795, ks:0.49795137064239775
    # dann_task_id = '20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625414259'

    # all4000pos004v4, # param = 20235
    # SOURCE: acc:0.9242250287026407, auc:0.9271533585860916, ks:0.702334481438959
    # TARGET: acc:0.681469938968965, auc:0.7825145547460837, ks:0.4484242613706949
    # DA:     acc:0.6565381119335151, auc:0.8153084812963785, ks:0.48491793234536423
    # dann_task_id = '20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625444243'

    # all4000pos004v4, # param = 20235
    # SOURCE: acc:0.9243628013777268, auc:0.929361977536147, ks:0.7047837734404898
    # TARGET: acc:0.6795221399818205, auc:0.791092224190332, ks:0.45565286099473856
    # DA:     acc:0.6564082586677055, auc:0.812296777425825, ks:0.48593116547034493
    # FT:     acc:0.6570575249967536, auc:0.812698410846858, ks:0.48667085743814303:
    # dann_task_id = '20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625452322'

    # all4000pos004v4, # param = 20235
    # SOURCE: acc:0.9256946039035592, auc:0.9286786650222056, ks:0.7013904834800357
    # TARGET: acc:0.6809505259057266, auc:0.7948005713999093, ks:0.4574100163248785
    # DA:     acc:0.6552395792754188, auc:0.8135058779477662, ks:0.4901208676739993
    # FT:     acc:0.6558888456044669, auc:0.8136937904804786, ks:0.4895827561690718
    # dann_task_id = '20210712_src_lr0.0006_bs128_pw1.0_me600_ts1626063288'

    # all4000pos004v4, using_interaction = True
    # SOURCE: acc:0.9244546498277841, auc:0.9281712342640766, ks:0.7042990177318535
    # TARGET: acc:0.6754966887417219, auc:0.7942495557171778, ks:0.4668313784111501
    # DA: acc:0.6556291390728477, auc:0.8131398503756938, ks:0.4875148799022023
    # FT: acc:0.6561485521360862, auc:0.8131544678200425, ks:0.4886377620860384
    # dann_task_id = '20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625426396'

    # all4000pos004v4, using_interaction = True
    # [INFO] SOURCE: acc:0.9236969001148105, auc:0.9276232895401119, ks:0.7018497257303227
    # [INFO] TARGET: acc:0.6741981560836255, auc:0.8012153329627818, ks:0.4764190367096884
    # DA: acc:0.6584859109206597, auc:0.8138231521071575, ks:0.49077387739485495
    # FT: acc:0.6584859109206597, auc:0.8103538967117531, ks:0.48272961708565637
    # dann_task_id = '20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625385790'

    # all4000pos004v4, using_interaction = True
    # [INFO] SOURCE: acc:0.9242250287026407, auc:0.9258253318177577, ks:0.6991963260619977
    # [INFO] TARGET: acc:0.6764056616023894, auc:0.7927522336621444, ks:0.4619204274446928
    # DA: acc:0.6582262043890403, auc:0.8142379787012536, ks:0.4974249968359248
    # FT: acc:0.6564082586677055, auc:0.812031877256044, ks:0.48955672909858555
    # dann_task_id = '20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625462494'

    # all4000pos004v4, using_interaction = True
    # SOURCE: acc:0.9253042479908151, auc:0.9273876225162275, ks:0.701696644980227
    # TARGET: acc:0.6796519932476301, auc:0.805552376089273, ks:0.4749727481261968
    # DA:     acc:0.6577067913258019, auc:0.8148477032896467, ks:0.4911215611459726
    # FT:     acc:0.6582262043890403, auc:0.8143626123538328, ks:0.490875434003223
    # dann_task_id = '20210706_src_lr0.0006_bs128_pw1.0_me600_ts1625473672'

    # all4000pos004v4, using_interaction = True
    # SOURCE: acc:0.9258553386911595, auc:0.9288261568310234, ks:0.7078964153591019
    # TARGET: acc:0.6782236073237242, auc:0.7959575001997596, ks:0.46002045129919555
    # DA:     acc:0.6566679651993248, auc:0.8146424571654202, ks:0.49703510111334626
    # FT:     acc:0.6575769380599922, auc:0.8149363952055366, ks:0.4938595797991439
    # dann_task_id = '20210712_src_lr0.0006_bs128_pw1.0_me600_ts1626060178'

    # Hyper-parameters
    # weight_decay = 0.00001
    weight_decay = 0.0
    batch_size = 128
    lr = 8e-4
    # lr = 5e-4
    # lr = 3e-4
    # batch_size = 32; lr = 5e-4;
    # batch_size = 32; lr = 8e-4;
    # batch_size = 64; lr = 8e-4;
    # batch_size = 128; lr = 8e-4;
    pos_class_weight = 1.0
    load_global_classifier = False
    using_interaction = True
    timestamp = get_timestamp()

    dann_root_folder = "census_dann"

    # Load models
    model = create_global_model(pos_class_weight=pos_class_weight, using_interaction=using_interaction)

    # load pre-trained model
    model.load_model(root=dann_root_folder,
                     task_id=dann_task_id,
                     load_global_classifier=load_global_classifier,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    # target_train_file_name = data_dir + 'grad_census9495_ft1623962998_train.csv'
    # target_test_file_name = data_dir + 'grad_census9495_ft1623963001_test.csv'
    # target_train_file_name = data_dir + 'grad_census9495_da_300_train.csv'
    # target_test_file_name = data_dir + 'grad_census9495_da_300_test.csv'
    # tag = "all4000pos004v4"
    tag = "all4000pos001"
    # tag = "all4000pos001v2"
    # tag = "all4000pos002"
    target_train_file_name = data_dir + f'grad_census9495_ft_{tag}_train.csv'
    target_test_file_name = data_dir + f'grad_census9495_ft_{tag}_test.csv'

    print(f"[INFO] load target train data from {target_train_file_name}.")
    print(f"[INFO] load target test data from {target_test_file_name}.")

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
                                         patience=900,
                                         max_global_epochs=500)
    plat_target.set_model_save_info("census_target")
    glr = "ft_glr" if load_global_classifier else "rt_glr"
    # appendix = "bs" + str(batch_size) + "_lr" + str(lr) + "_pw" + str(pos_class_weight) + "_ts" + str(timestamp)

    hyperparameter_dict = {"lr": lr, "bs": batch_size, "pw": pos_class_weight, "ts": timestamp}
    appendix = create_id_from_hyperparameters(hyperparameter_dict)
    target_task_id = dann_task_id + "@target_" + glr + "_" + appendix
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

    acc, auc, ks = test_classifier(model, target_valid_loader, 'test')
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
