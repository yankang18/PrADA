from sklearn import preprocessing

from data_process.census_process.mapping_resource import education_value_map, workclass_index_map, education_index_map, \
    marital_status_index_map, occupation_index_map, country_index_map, race_index_map, relationship_index_map, \
    income_label_index, gender_index_map
from data_process.census_process.utils import bucketized_age


def assign_native_country_identifier(native_country):
    # native_country_index_map = {
    #     "None": 0,
    #     "United-States": 1, "Vietnam": 2, "Columbia": 3, "Mexico": 4, "Peru": 5,
    #     "Cuba": 6, "Philippines": 7, "Dominican-Republic": 8,
    #     "El-Salvador": 9, "Canada": 10, "Scotland": 11, "Portugal": 12,
    #     "Guatemala": 13, "Ecuador": 14, "Germany": 15,
    #     "Outlying-US(Guam-USVI-etc)": 16, "Puerto-Rico": 17, "Italy": 18, "China": 19, "Poland": 20,
    #     "Nicaragua": 21, "Taiwan": 22, "England": 23, "Ireland": 24, "South-Korea": 25, "Trinadad&Tobago": 26,
    #     "Jamaica": 27, "Honduras": 28, "Iran": 29, "Hungary": 30, "France": 31, "Cambodia": 32,
    #     "India": 33, "Hong-Kong": 34, "Japan": 35, "Haiti": 36, "Holand-Netherlands": 37, "Greece": 38,
    #     "Thailand": 39, "Panama": 40, "Yugoslavia": 41, "Laos": 42}
    asian_country = ["Vietnam", "Philippines", "China", "Taiwan", "South-Korea", "India", "Hong-Kong", "Japan",
                     "Thailand", "Laos", "Cambodia"]
    if native_country in asian_country:
        return 1
    else:
        return 0


def assign_doctorate_identifier(degree):
    # post_grads = ["Doctorate", "Masters"]

    doctorate = ["Doctorate", "Masters"]
    # doctorate = ["Doctorate"]
    exclusive = ["Bachelors", "Prof-school"]
    if degree in doctorate:
        return 1
    elif degree in exclusive:
        return -1
    else:
        return 0


def process_census_data(data_frame):
    data_frame = data_frame.dropna()
    data_frame = data_frame.replace(to_replace="?", value="None")
    data_frame = data_frame.replace(to_replace="South", value="South-Korea")
    data_frame = data_frame.replace(to_replace="Hong", value="Hong-Kong")
    data_frame = data_frame.replace(to_replace="<=50K.", value="<=50K")
    data_frame = data_frame.replace(to_replace=">50K.", value=">50K")

    data_frame['education_year'] = data_frame.apply(lambda row: education_value_map[row.education], axis=1)
    data_frame['age_bucket'] = data_frame.apply(lambda row: bucketized_age(row.age), axis=1)
    data_frame['is_asian'] = data_frame.apply(lambda row: assign_native_country_identifier(row.native_country), axis=1)
    data_frame['post_graduate'] = data_frame.apply(lambda row: assign_doctorate_identifier(row.education), axis=1)
    data_frame = data_frame.replace({"workclass": workclass_index_map, "education": education_index_map,
                                     "marital_status": marital_status_index_map, "occupation": occupation_index_map,
                                     "native_country": country_index_map, "race": race_index_map,
                                     "relationship": relationship_index_map, "income_label": income_label_index,
                                     "gender": gender_index_map})
    return data_frame


# continuous_cols = ["age", "education_year", "capital_gain", "capital_loss", "gender", "hours_per_week"]
continuous_cols = ["age", "education_year", "capital_gain", "capital_loss", "hours_per_week"]


def standardize_census_data(data_frame):
    feat = data_frame[continuous_cols].values
    scaler = preprocessing.StandardScaler()
    s_feat = scaler.fit_transform(feat)
    for idx in range(len(continuous_cols)):
        data_frame[continuous_cols[idx]] = s_feat[:, idx]
    return data_frame
