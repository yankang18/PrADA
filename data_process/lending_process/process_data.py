import pandas as pd
import numpy as np

# target_map = {'Good Loan': 0, 'Bad Loan': 1}
target_map = {'Good Loan': 1, 'Bad Loan': 0}

grade_map = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
sub_grade = ['C1' 'D2' 'D1' 'C4' 'C3' 'C2' 'D5' 'B3' 'A4' 'B5' 'C5' 'D4' 'E1' 'E4'
             'B4' 'D3' 'A1' 'E5' 'B2' 'B1' 'A5' 'F5' 'A3' 'E3' 'A2' 'E2' 'F4' 'G1'
             'G2' 'F1' 'F2' 'F3' 'G4' 'G3' 'G5']

# emp_length_map = {np.nan: 0, '< 1 year': 1, '1 year': 2, '2 years': 2, '3 years': 2, '4 years': 3, '5 years': 3,
#                   '6 years': 3, '7 years': 4, '8 years': 4, '9 years': 4, '10+ years': 5}
emp_length_map = {np.nan: 0, '< 1 year': 1, '1 year': 2, '2 years': 3, '3 years': 4, '4 years': 5, '5 years': 6,
                  '6 years': 7, '7 years': 8, '8 years': 9, '9 years': 10, '10+ years': 11}

home_ownership_map = {'ANY': 0, 'NONE': 0, 'OTHER': 0, 'MORTGAGE': 1, 'RENT': 2, 'OWN': 3}

# verification_status_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
verification_status_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 1}
# verification_status_joint = [nan 'Verified' 'Not Verified' 'Source Verified']

# The number of payments on the loan. Values are in months and can be either 36 or 60.
term_map = {' 36 months': 0, ' 60 months': 1}
initial_list_status_map = {'w': 0, 'f': 1}
purpose_map = {'debt_consolidation': 0, 'credit_card': 0, 'small_business': 1, 'educational': 2,
               'car': 3, 'other': 3, 'vacation': 3, 'house': 3, 'home_improvement': 3, 'major_purchase': 3,
               'medical': 3, 'renewable_energy': 3, 'moving': 3, 'wedding': 3}

# Indicates whether the loan is an individual application or a joint application with two co-borrowers
application_type_map = {'Individual': 0, 'Joint App': 1}
disbursement_method_map = {'Cash': 0, 'DirectPay': 1}


def loan_condition(status):
    bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period",
                "Late (16-30 days)", "Late (31-120 days)"]
    if status in bad_loan:
        return 'Bad Loan'
    else:
        return 'Good Loan'


def compute_annual_income(row):
    if row['verification_status'] == row['verification_status_joint']:
        return row['annual_inc_joint']
    return row['annual_inc']


def determine_good_bad_loan(df_loan):
    # Determining the loans that are bad from loan_status column
    print("==> determine_good_bad_loan")

    df_loan['target'] = np.nan
    df_loan['target'] = df_loan['loan_status'].apply(loan_condition)
    return df_loan


def determine_annual_income(df_loan):
    print("==> determine_annual_income")

    df_loan['annual_inc_comp'] = np.nan
    df_loan['annual_inc_comp'] = df_loan.apply(compute_annual_income, axis=1)
    return df_loan


def determine_issue_year(df_loan):
    print("==> determine_issue_year")

    # transform the issue dates by year
    dt_series = pd.to_datetime(df_loan['issue_d'])
    df_loan['issue_year'] = dt_series.dt.year
    return df_loan


def digitize_columns(data_frame):
    print("==> digitize_columns")

    # data_frame = data_frame.replace(to_replace=np.nan, value="None")
    data_frame = data_frame.replace({"target": target_map, "grade": grade_map, "emp_length": emp_length_map,
                                     "home_ownership": home_ownership_map,
                                     "verification_status": verification_status_map,
                                     "term": term_map, "initial_list_status": initial_list_status_map,
                                     "purpose": purpose_map, "application_type": application_type_map,
                                     "disbursement_method": disbursement_method_map})
    return data_frame


if __name__ == "__main__":
    dir = "../../../data/lending_club_bundle_archive/"
    to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/"
    file_path = dir + "loan.csv"
    df_loan = pd.read_csv(file_path, low_memory=False)
    print(f" ==> load data loan table with shape:{df_loan.shape} to :{file_path}")

    df_loan = determine_good_bad_loan(df_loan)
    df_loan = determine_annual_income(df_loan)
    df_loan = determine_issue_year(df_loan)
    df_loan = digitize_columns(df_loan)

    # file_path = to_dir + "loan_processed.csv"
    # df_loan.to_csv(file_path, index=False)
    # print(f" ==> save processed data loan table with shape:{df_loan.shape} to :{file_path}")

    df_loan_2018 = df_loan[df_loan['issue_year'] == 2018]
    num_good = np.sum(df_loan_2018['target'].values)
    num_bad = df_loan_2018.shape[0] - num_good
    print(f"df_loan_2018 shape:{df_loan_2018.shape} with # good:{num_good} and # bad:{num_bad}")
    # file_path = to_dir + "loan_processed_2018.csv"
    # df_loan_2018.to_csv(file_path, index=False)
    # print(f" ==> save processed 2018 data loan table with shape:{df_loan_2018.shape} to :{file_path}")

    df_loan_15_17 = df_loan[(df_loan['issue_year'] >= 2015) & (df_loan['issue_year'] <= 2017)]
    num_good = np.sum(df_loan_15_17['target'].values)
    num_bad = df_loan_15_17.shape[0] - num_good
    print(f"df_loan_15_17 shape:{df_loan_15_17.shape} with # good:{num_good} and # bad:{num_bad}")
    # file_path = to_dir + "loan_processed_2015_17.csv"
    # df_loan_15_17.to_csv(file_path, index=False)
    # print(f" ==> save processed 2015-2018 data loan table with shape:{df_loan_15_17.shape} to :{file_path}")

    df_loan_15_16 = df_loan[(df_loan['issue_year'] >= 2015) & (df_loan['issue_year'] <= 2016)]
    num_good = np.sum(df_loan_15_16['target'].values)
    num_bad = df_loan_15_16.shape[0] - num_good
    print(f"df_loan_15_16 shape:{df_loan_15_16.shape} with # good:{num_good} and # bad:{num_bad}")
    # file_path = to_dir + "loan_processed_2015_16.csv"
    # df_loan_15_16.to_csv(file_path, index=False)
    # print(f" ==> save processed 2015-2016 data loan table with shape:{df_loan_15_16.shape} to :{file_path}")

    df_loan_16_17 = df_loan[(df_loan['issue_year'] >= 2016) & (df_loan['issue_year'] <= 2017)]
    num_good = np.sum(df_loan_16_17['target'].values)
    num_bad = df_loan_16_17.shape[0] - num_good
    print(f"df_loan_16_17 shape:{df_loan_16_17.shape} with # good:{num_good} and # bad:{num_bad}")
    # file_path = to_dir + "loan_processed_2016_17.csv"
    # df_loan_16_17.to_csv(file_path, index=False)
    # print(f" ==> save processed 2016-2017 data loan table with shape:{df_loan_16_17.shape} to :{file_path}")
