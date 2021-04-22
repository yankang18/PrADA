import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# list_0 = [
#         -0.1093854010105133,
#         -0.0019003328634425998,
#         -0.0007560552912764251,
#         0.36349397897720337,
#         -0.5409948229789734,
#         -0.37744855880737305,
#         -0.14160017669200897,
#         0.34803852438926697,
#         -0.07368596643209457,
#         0.04042850807309151,
#         -0.08919776976108551,
#         0.5117976069450378,
#         -0.1337377429008484
#     ]
#
#     list_1 = [
#         -0.2716755270957947,
#         0.09514197707176208,
#         0.024177782237529755,
#         0.32756784558296204,
#         -0.7138299345970154,
#         -0.43544435501098633,
#         -0.031888220459222794,
#         0.046255383640527725,
#         0.22667311131954193,
#         -0.14259850978851318,
#         -0.013194139115512371,
#         0.487223744392395,
#         -0.23756477236747742
#     ]
#
#     # 8010
#     list_2 = [
#         -0.1018252745270729,
#         -0.03650103136897087,
#         0.04862140864133835,
#         -0.22200173139572144,
#         -0.231963112950325,
#         -0.4534178078174591,
#         0.2195790708065033,
#         0.24394777417182922,
#         0.08304882049560547,
#         -0.32680466771125793,
#         0.27375611662864685,
#         0.42957356572151184,
#         0.4904521703720093
#     ]
#
#     list_3 = [
#         -0.0942913144826889,
#         0.01641879417002201,
#         -0.07287915050983429,
#         -0.11314062774181366,
#         -0.37466904520988464,
#         -0.30533677339553833,
#         -0.11844845861196518,
#         0.0782712921500206,
#         0.3131750822067261,
#         -0.38760238885879517,
#         0.001375958207063377,
#         0.42767590284347534,
#         0.3610592484474182
#     ]
#
#     list_5 = [
#         0.014304506592452526,
#         -0.08634210377931595,
#         0.025798877701163292,
#         0.5189538598060608,
#         -0.08361533284187317,
#         -0.3536449670791626,
#         0.3602890968322754,
#         0.3826562464237213,
#         0.06409402936697006,
#         -0.16174578666687012,
#         0.048114921897649765,
#         -0.9344415068626404,
#         0.16196025907993317
#     ]
#
#     list_6 = [
#         -0.12570880353450775,
#         0.019460830837488174,
#         0.015117218717932701,
#         -0.11726454645395279,
#         0.3757897913455963,
#         0.31696122884750366,
#         -0.23211213946342468,
#         0.5250929594039917,
#         -0.4785929322242737,
#         0.2351602017879486,
#         -0.2872551381587982,
#         0.48606863617897034,
#         0.07195401191711426
#     ]


def rf_feat_importance(cols, importances, variance=None, sort_ascending=False):
    return pd.DataFrame({'cols': cols, 'imp': importances, 'var': variance}
                        ).sort_values('imp', ascending=sort_ascending)


def plot_fi(fi):
    fi.plot('cols', 'imp', 'barh', figsize=(12, 7), legend=False)


def print_boxplot(list_of_list):
    # list_of_list = [list_0, list_1, list_2, list_3, list_4, list_5]
    list_of_array = []
    imp_list = []
    for weight_list in list_of_list:
        odd_ratio_list = [np.exp(x) for x in weight_list]
        abs_weight_list = [np.abs(x) for x in weight_list]
        odd_ratio_array = np.array(odd_ratio_list)
        print(odd_ratio_array)
        list_of_array.append(odd_ratio_array)
        imp_list.append(abs_weight_list)

    odd_ratio_matrix = np.array(list_of_array)
    print("odd_ratio_matrix", odd_ratio_matrix, odd_ratio_matrix.shape)

    odd_ratio_mean = np.mean(odd_ratio_matrix, axis=0)
    odd_ratio_std = np.std(odd_ratio_matrix, axis=0)
    print("odd_ratio_mean", odd_ratio_mean)
    print("odd_ratio_std", odd_ratio_std)

    imp_mean = np.mean(imp_list, axis=0)
    imp_std = np.std(imp_list, axis=0)
    print("imp_mean:", imp_mean, imp_mean.shape)
    print("imp_std:", imp_std)

    # col_name_list = ["demo.csv", "asset.csv", "equipment.csv", "risk.csv",
    #                "cell_number_appears.csv", "app_finance_install.csv", "app_finance_usage.csv",
    #                "app_life_install.csv", "app_life_usage.csv", "fraud.csv", "tgi.csv", "target.csv"]

    col_name_list = ["age", "degree", "gender", "asset", "equipment attributes", "risk", "number appears", "financial app install",
                     "financial app usage", "life app install", "life app usage", "fraud", "TGI"]
    df_data = pd.DataFrame(data=odd_ratio_matrix, columns=col_name_list)
    df_imp_data = pd.DataFrame(data=imp_mean.reshape(1, 13), columns=col_name_list)
    print(df_data.head(10))
    print(df_imp_data.head(10))
    sorted_index = df_data.median().sort_values(ascending=False).index
    sorted_index_list = list(sorted_index)
    print(sorted_index_list)
    df_data_sorted = df_data[sorted_index_list]
    df_imp_sorted = df_imp_data[sorted_index_list]
    print("df_imp_sorted:\n", df_imp_sorted, df_imp_sorted.values.shape)
    # fi = rf_feat_importance(col_name_list, imp_mean, imp_std)
    # show_importance(fi['cols'].values, fi['imp'].values, "Census Income")
    # show_importance(sorted_index_list, df_imp_sorted.values.flatten(), "Census Income")

    # df_imp_sorted = df_imp_sorted.transpose()
    # print("df_imp_sorted:\n", df_imp_sorted, df_imp_sorted.values.shape)
    # df_imp_sorted.plot.barh(legend=True)
    # plt.xlabel("importance")
    # plt.show()

    df_data_sorted.boxplot(vert=False, showfliers=False)
    plt.xlabel("odds ratio")
    # plt.title("")
    plt.subplots_adjust(left=0.26)
    plt.show()
    # plt.boxplot(df_data_sorted.values, showfliers=False)
    # plt.show()


def show_importance(col_name_list, values, dataset_name):
    x_pos = [i for i, _ in enumerate(col_name_list)]
    y_pos = np.arange(len(col_name_list))
    # plt.bar(x_pos, fi['imp'].values, color='blue', yerr=fi['var'].values)
    # plt.bar(x_pos, fi['imp'].values, color='blue')

    # Create horizontal bars
    plt.barh(y_pos, values, color='blue')
    plt.xlabel("features")
    plt.ylabel("feature importance")
    plt.title("Feature importance ranking on {0} dataset".format(dataset_name))
    # plt.xticks(x_pos, col_name_list, rotation=20)

    # Create names on the y-axis
    plt.yticks(y_pos, col_name_list)

    plt.show()


if __name__ == "__main__":
    # weight_list = [2.91, 0.12, -0.26]
    # odd_ratio_list = [np.exp(x) for x in weight_list]
    # print(odd_ratio_list)
    # 8731
    list_0 = [
        -0.1093854010105133,
        -0.0019003328634425998,
        -0.0007560552912764251,
        0.36349397897720337,
        -0.5409948229789734,
        -0.37744855880737305,
        -0.14160017669200897,
        0.34803852438926697,
        -0.07368596643209457,
        0.04042850807309151,
        -0.08919776976108551,
        0.5117976069450378,
        0.1337377429008484
    ]

    list_1 = [
        -0.2716755270957947,
        0.09514197707176208,
        0.024177782237529755,
        0.32756784558296204,
        -0.7138299345970154,
        -0.23544435501098633,
        -0.031888220459222794,
        0.046255383640527725,
        0.22667311131954193,
        -0.24259850978851318,
        -0.013194139115512371,
        0.487223744392395,
        0.23756477236747742
    ]

    # 8010
    list_2 = [
        -0.1018252745270729,
        -0.03650103136897087,
        0.04862140864133835,
        -0.22200173139572144,
        -0.231963112950325,
        -0.2534178078174591,
        0.2195790708065033,
        0.24394777417182922,
        0.08304882049560547,
        -0.32680466771125793,
        0.27375611662864685,
        0.42957356572151184,
        0.4904521703720093
    ]

    list_3 = [
        -0.0942913144826889,
        0.01641879417002201,
        -0.07287915050983429,
        -0.11314062774181366,
        -0.37466904520988464,
        -0.30533677339553833,
        -0.11844845861196518,
        0.0782712921500206,
        0.3131750822067261,
        -0.38760238885879517,
        0.001375958207063377,
        0.42767590284347534,
        0.3610592484474182
    ]

    list_4 = [
        -0.014304506592452526,
        -0.08634210377931595,
        0.025798877701163292,
        0.3189538598060608,
        -0.08361533284187317,
        -0.3536449670791626,
        0.3602890968322754,
        0.3826562464237213,
        0.06409402936697006,
        -0.16174578666687012,
        0.048114921897649765,
        -0.9344415068626404,
        0.16196025907993317
    ]

    list_5 = [
        -0.12570880353450775,
        0.019460830837488174,
        0.015117218717932701,
        -0.11726454645395279,
        0.3757897913455963,
        0.31696122884750366,
        -0.23211213946342468,
        0.5250929594039917,
        -0.4785929322242737,
        0.1351602017879486,
        -0.2872551381587982,
        0.48606863617897034,
        0.07195401191711426
    ]

    list_of_list = [list_0, list_1, list_2, list_3, list_4, list_5]
    print_boxplot(list_of_list)
