import pandas as pd
from sklearn import manifold

from utils import draw_distribution


def draw_distribution_for_each_group(dim_reducer,
                                     feature_group_name_list,
                                     tag_list,
                                     num_samples,
                                     print_num_points,
                                     data_dir,
                                     to_dir,
                                     version="0"):
    for name in feature_group_name_list:
        for tag in tag_list:
            print("-" * 100)
            src_file_name = 'prada_{}{}_src_emb_{}_v{}.csv'.format(tag, num_samples, name, version)
            tgt_file_name = 'prada_{}{}_tgt_emb_{}_v{}.csv'.format(tag, num_samples, name, version)

            print(f"[INFO] tag:{tag}; feature_group_name:{name}")

            df_src_data = pd.read_csv(data_dir + src_file_name, header=None, skipinitialspace=True)
            df_tgt_data = pd.read_csv(data_dir + tgt_file_name, header=None, skipinitialspace=True)

            print(f"[INFO] df_src_data shape :{df_src_data.shape}")
            print(f"[INFO] df_tgt_data shape :{df_tgt_data.shape}")

            draw_distribution(df_src_data,
                              df_tgt_data,
                              num_points=print_num_points,
                              dim_reducer=dim_reducer,
                              tag=tag,
                              feature_group_name=name,
                              to_dir=to_dir,
                              version=version)


def draw_distribution_for_concat_feature(dim_reducer,
                                         feature_group_name_list,
                                         tag,
                                         num_samples,
                                         print_num_points,
                                         data_dir,
                                         to_dir):
    df_src_data_list = list()
    df_tgt_data_list = list()
    for name in feature_group_name_list:
        src_file_name = 'prada_{}{}_src_emb_{}.csv'.format(tag, num_samples, name)
        tgt_file_name = 'prada_{}{}_tgt_emb_{}.csv'.format(tag, num_samples, name)

        print(f"[INFO] tag:{tag}; feature_group_name:{name}")

        df_src_data = pd.read_csv(data_dir + src_file_name, header=None, skipinitialspace=True)
        df_tgt_data = pd.read_csv(data_dir + tgt_file_name, header=None, skipinitialspace=True)

        df_src_data_list.append(df_src_data)
        df_tgt_data_list.append(df_tgt_data)
        print(f"[INFO] df_src_data shape :{df_src_data.shape}")
        print(f"[INFO] df_tgt_data shape :{df_tgt_data.shape}")

    df_src_all_data = pd.concat(df_src_data_list, axis=1)
    df_tgt_all_data = pd.concat(df_tgt_data_list, axis=1)

    print(f"[INFO] df_src_all_data shape :{df_src_all_data.shape}")
    print(f"[INFO] df_tgt_all_data shape :{df_tgt_all_data.shape}")

    draw_distribution(df_src_all_data, df_tgt_all_data,
                      num_points=print_num_points, dim_reducer=dim_reducer, tag=tag,
                      feature_group_name='all_combined', to_dir=to_dir)


if __name__ == "__main__":
    # save df_src_data to /Users/yankang/Documents/Data/census/output//prada_dann_src_emb_employment.csv
    # save df_tgt_data to /Users/yankang/Documents/Data/census/output//prada_dann_tgt_emb_employment.csv
    # save df_src_data to /Users/yankang/Documents/Data/census/output//prada_dann_src_emb_demographics.csv
    # save df_tgt_data to /Users/yankang/Documents/Data/census/output//prada_dann_tgt_emb_demographics.csv
    # save df_src_data to /Users/yankang/Documents/Data/census/output//prada_dann_src_emb_residence.csv
    # save df_tgt_data to /Users/yankang/Documents/Data/census/output//prada_dann_tgt_emb_residence.csv
    # save df_src_data to /Users/yankang/Documents/Data/census/output//prada_dann_src_emb_household.csv
    # save df_tgt_data to /Users/yankang/Documents/Data/census/output//prada_dann_tgt_emb_household.csv

    feature_group_name_list = ['employment', 'household', 'demographics', 'migration']
    # feature_group_name_list = ['household', 'demographics', 'migration']
    # tag_list = ['no_dann', 'dann']
    tag_list = ['dann']
    # tag_list = ['no_dann']
    num_samples = 4000
    print_num_points = 3000
    # print_num_points = 4000
    # version = 'ts1625444243'
    # version = 'ts1626078661'
    # version = 'ts1626108508'
    # version = 'ts1626060178'
    # version = 'ts1625473672'
    # version = 'ts1626063288'
    # version = 'ts1625385790'
    # version = 'ts1626161175'
    # version = 'ts1626112390'
    version = 'ts1624919877'
    data_dir = "/Users/yankang/Documents/Data/census/output_emb/"
    to_dir = "/Users/yankang/Documents/Data/census/distribution_fig_v10/"
    dim_reducer = manifold.TSNE(n_components=2, init='random', perplexity=10, learning_rate=600, n_iter=1000,
                                verbose=2, early_exaggeration=12, random_state=10)
    draw_distribution_for_each_group(dim_reducer,
                                     feature_group_name_list,
                                     tag_list,
                                     num_samples,
                                     print_num_points,
                                     data_dir,
                                     to_dir,
                                     version=version)

