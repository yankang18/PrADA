import pandas as pd
from sklearn.utils import shuffle


def create_train_and_test(df_data, df_datetime, num_train, to_dir):
    df_data_2014 = df_data[df_datetime['ListingInfo_Year'] == 2014]
    df_datetime_2014 = df_datetime[df_datetime['ListingInfo_Year'] == 2014]
    df_data_2014, df_datetime_2014 = shuffle(df_data_2014, df_datetime_2014)

    df_data_train = df_data_2014[:num_train]
    df_datetime_train = df_datetime_2014[:num_train]

    df_data_test = df_data_2014[num_train:]
    df_datetime_test = df_datetime_2014[num_train:]

    print(f"[INFO] df_data_train with shape: {df_data_train.shape}")
    print(f"[INFO] df_data_test with shape: {df_data_test.shape}")
    print(f"[INFO] df_datetime_train with shape: {df_datetime_train.shape}")
    print(f"[INFO] df_datetime_test with shape: {df_datetime_test.shape}")

    title = "PPD"
    tag = '2014'
    df_data_train.to_csv("{}/{}_data_{}_{}_train.csv".format(to_dir, title, tag, str(num_train)), index=False)
    df_data_test.to_csv("{}/{}_data_{}_{}_test.csv".format(to_dir, title, tag, str(num_train)), index=False)
    df_datetime_train.to_csv("{}/{}_datetime_{}_{}_train.csv".format(to_dir, title, tag, str(num_train)), index=False)
    df_datetime_test.to_csv("{}/{}_datetime_{}_{}_test.csv".format(to_dir, title, tag, str(num_train)), index=False)


if __name__ == "__main__":
    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output/"

    timestamp = '1620085151'
    data_all = data_dir + 'PPD_data_all_{}.csv'.format(timestamp)
    data_datetime = data_dir + 'PPD_data_datetime_{}.csv'.format(timestamp)

    df_data_all = pd.read_csv(data_all, skipinitialspace=True)
    df_data_datetime = pd.read_csv(data_datetime, skipinitialspace=True)

    print(f"[INFO] df_data_all: {df_data_all.shape}")
    print(f"[INFO] df_data_datetime: {df_data_datetime.shape}")
    print("2015:", df_data_all[df_data_datetime['ListingInfo_Year'] == 2015].shape)
    print("2014:", df_data_all[df_data_datetime['ListingInfo_Year'] == 2014].shape)
    print("2013:", df_data_all[df_data_datetime['ListingInfo_Year'] == 2013].shape)
    print("2012:", df_data_all[df_data_datetime['ListingInfo_Year'] == 2012].shape)

    to_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{timestamp}/"

    num_train = 55000
    create_train_and_test(df_data_all, df_data_datetime, num_train, to_dir)
