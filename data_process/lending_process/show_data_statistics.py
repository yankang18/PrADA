import pandas as pd

if __name__ == "__main__":
    dir = "../../../data/lending_club_bundle_archive/loan_data_v2/"

    file_path = dir + "loan_processed_2015_16.csv"
    df_loan = pd.read_csv(file_path, low_memory=False)

    print(f"loan_processed_2015_16 table shape:{df_loan.shape}")

    file_path = dir + "loan_processed_2016_17.csv"
    df_loan = pd.read_csv(file_path, low_memory=False)

    print(f"loan_processed_2016_17 table shape:{df_loan.shape}")

    file_path = dir + "loan_processed_2018.csv"
    df_loan = pd.read_csv(file_path, low_memory=False)

    print(f"loan_processed_2018 table shape:{df_loan.shape}")
