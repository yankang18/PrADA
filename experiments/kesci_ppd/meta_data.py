# 2 indicates wide features of active party (with Y) and 1 indicates feature groups of passive party
group_ind_list = [2, 1, 1, 1, 1, 1]
column_name_list = ['UserInfo_10',
                    'UserInfo_14',
                    'UserInfo_15',
                    'UserInfo_18',
                    'UserInfo_13',
                    'UserInfo_12',
                    'UserInfo_22',
                    'UserInfo_17',
                    'UserInfo_21',
                    'UserInfo_5',
                    'UserInfo_3',
                    'UserInfo_11',
                    'UserInfo_16',
                    'UserInfo_1',
                    'UserInfo_6',
                    'UserInfo_23',
                    'UserInfo_9',
                    'UserInfo_20_Longitude',
                    'UserInfo_20_Latitude',
                    'UserInfo_20_LongpLat',
                    'UserInfo_20_LongmLat',
                    'UserInfo_20_CityRank',
                    'UserInfo_CityRank_median',
                    'UserInfo_Latitude_median',
                    'UserInfo_Longitude_median',
                    'UserInfo_LongmLat_median',
                    'UserInfo_LongpLat_median',
                    'UserInfo_CityRank_std',
                    'UserInfo_Latitude_std',
                    'UserInfo_Longitude_std',
                    'UserInfo_LongmLat_std',
                    'UserInfo_LongpLat_std',
                    'ThirdParty_Info_1_median',
                    'ThirdParty_Info_10_median',
                    'ThirdParty_Info_11_median',
                    'ThirdParty_Info_12_median',
                    'ThirdParty_Info_13_median',
                    'ThirdParty_Info_14_median',
                    'ThirdParty_Info_15_median',
                    'ThirdParty_Info_16_median',
                    'ThirdParty_Info_17_median',
                    'ThirdParty_Info_2_median',
                    'ThirdParty_Info_3_median',
                    'ThirdParty_Info_4_median',
                    'ThirdParty_Info_5_median',
                    'ThirdParty_Info_6_median',
                    'ThirdParty_Info_7_median',
                    'ThirdParty_Info_8_median',
                    'ThirdParty_Info_9_median',
                    'ThirdParty_Info_1_std',
                    'ThirdParty_Info_10_std',
                    'ThirdParty_Info_11_std',
                    'ThirdParty_Info_12_std',
                    'ThirdParty_Info_13_std',
                    'ThirdParty_Info_14_std',
                    'ThirdParty_Info_15_std',
                    'ThirdParty_Info_16_std',
                    'ThirdParty_Info_17_std',
                    'ThirdParty_Info_2_std',
                    'ThirdParty_Info_3_std',
                    'ThirdParty_Info_4_std',
                    'ThirdParty_Info_5_std',
                    'ThirdParty_Info_6_std',
                    'ThirdParty_Info_7_std',
                    'ThirdParty_Info_8_std',
                    'ThirdParty_Info_9_std',
                    'ThirdParty_Info_1_min',
                    'ThirdParty_Info_10_min',
                    'ThirdParty_Info_11_min',
                    'ThirdParty_Info_12_min',
                    'ThirdParty_Info_13_min',
                    'ThirdParty_Info_14_min',
                    'ThirdParty_Info_15_min',
                    'ThirdParty_Info_16_min',
                    'ThirdParty_Info_17_min',
                    'ThirdParty_Info_2_min',
                    'ThirdParty_Info_3_min',
                    'ThirdParty_Info_4_min',
                    'ThirdParty_Info_5_min',
                    'ThirdParty_Info_6_min',
                    'ThirdParty_Info_7_min',
                    'ThirdParty_Info_8_min',
                    'ThirdParty_Info_9_min',
                    'ThirdParty_Info_1_max',
                    'ThirdParty_Info_10_max',
                    'ThirdParty_Info_11_max',
                    'ThirdParty_Info_12_max',
                    'ThirdParty_Info_13_max',
                    'ThirdParty_Info_14_max',
                    'ThirdParty_Info_15_max',
                    'ThirdParty_Info_16_max',
                    'ThirdParty_Info_17_max',
                    'ThirdParty_Info_2_max',
                    'ThirdParty_Info_3_max',
                    'ThirdParty_Info_4_max',
                    'ThirdParty_Info_5_max',
                    'ThirdParty_Info_6_max',
                    'ThirdParty_Info_7_max',
                    'ThirdParty_Info_8_max',
                    'ThirdParty_Info_9_max',
                    'ThirdParty_Info_1_first',
                    'ThirdParty_Info_10_first',
                    'ThirdParty_Info_11_first',
                    'ThirdParty_Info_12_first',
                    'ThirdParty_Info_13_first',
                    'ThirdParty_Info_14_first',
                    'ThirdParty_Info_15_first',
                    'ThirdParty_Info_16_first',
                    'ThirdParty_Info_17_first',
                    'ThirdParty_Info_2_first',
                    'ThirdParty_Info_3_first',
                    'ThirdParty_Info_4_first',
                    'ThirdParty_Info_5_first',
                    'ThirdParty_Info_6_first',
                    'ThirdParty_Info_7_first',
                    'ThirdParty_Info_8_first',
                    'ThirdParty_Info_9_first',
                    'Education_Info3',
                    'Education_Info5',
                    'Education_Info1',
                    'Education_Info7',
                    'Education_Info6',
                    'Education_Info8',
                    'Education_Info2',
                    'Education_Info4',
                    'SocialNetwork_9',
                    'SocialNetwork_6',
                    'SocialNetwork_14',
                    'SocialNetwork_8',
                    'SocialNetwork_10',
                    'SocialNetwork_17',
                    'SocialNetwork_16',
                    'SocialNetwork_13',
                    'SocialNetwork_3',
                    'SocialNetwork_4',
                    'SocialNetwork_15',
                    'SocialNetwork_5',
                    'SocialNetwork_2',
                    'SocialNetwork_7',
                    'SocialNetwork_12',
                    'WeblogInfo_57',
                    'WeblogInfo_30',
                    'WeblogInfo_26',
                    'WeblogInfo_17',
                    'WeblogInfo_6',
                    'WeblogInfo_28',
                    'WeblogInfo_4',
                    'WeblogInfo_3',
                    'WeblogInfo_14',
                    'WeblogInfo_35',
                    'WeblogInfo_15',
                    'WeblogInfo_34',
                    'WeblogInfo_33',
                    'WeblogInfo_8',
                    'WeblogInfo_27',
                    'WeblogInfo_7',
                    'WeblogInfo_18',
                    'WeblogInfo_36',
                    'WeblogInfo_29',
                    'WeblogInfo_39',
                    'WeblogInfo_48',
                    'WeblogInfo_9',
                    'WeblogInfo_38',
                    'WeblogInfo_25',
                    'WeblogInfo_16',
                    'WeblogInfo_24',
                    'WeblogInfo_5',
                    'WeblogInfo_56',
                    'WeblogInfo_42',
                    'WeblogInfo_2',
                    'WeblogInfo_19',
                    'WeblogInfo_21',
                    'WeblogInfo_20']

# df_group_index = [(0, 4),
#                   (4, 13),
#                   (17, 15),
#                   (32, 85),
#                   (117, 8),
#                   (125, 12),
#                   (137, 3),
#                   (140, 30),
#                   (170, 3)]

# group_info = [(0, (17, False)),
#               (17, (15, False)),
#               (32, (85, False)),
#               (117, (8, True)),
#               (125, (12, False), (3, True)),
#               (140, (30, False), (3, True))]

# [start_index, (length, is_categorical), (length, is_categorical)]
group_info = [(0, (5, False)),
              (17, (15, False)),
              (32, (85, False)),
              (117, (8, True)),
              (125, (12, False), (3, True)),
              (140, (30, False), (3, True))]

embedding_shape_map = {'UserInfo_13': (3, 3),
                       'UserInfo_12': (3, 3),
                       'UserInfo_22': (9, 9),
                       'UserInfo_17': (2, 2),
                       'UserInfo_21': (2, 2),
                       'UserInfo_5': (3, 3),
                       'UserInfo_3': (9, 9),
                       'UserInfo_11': (3, 3),
                       'UserInfo_16': (6, 6),
                       'UserInfo_1': (9, 9),
                       'UserInfo_6': (3, 3),
                       'UserInfo_23': (31, 15),
                       'UserInfo_9': (4, 4),
                       'Education_Info3': (3, 2),
                       'Education_Info5': (2, 1),
                       'Education_Info1': (2, 1),
                       'Education_Info7': (2, 1),
                       'Education_Info6': (6, 5),
                       'Education_Info8': (7, 6),
                       'Education_Info2': (7, 6),
                       'Education_Info4': (6, 5),
                       'SocialNetwork_2': (3, 2),
                       'SocialNetwork_7': (3, 2),
                       'SocialNetwork_12': (3, 2),
                       'WeblogInfo_19': (8, 6),
                       'WeblogInfo_21': (5, 4),
                       'WeblogInfo_20': (45, 15)}

# embedding_shape_map = {'UserInfo_13': (3, 3),
#                        'UserInfo_12': (3, 3),
#                        'UserInfo_22': (9, 9),
#                        'UserInfo_17': (2, 2),
#                        'UserInfo_21': (2, 2),
#                        'UserInfo_5': (3, 3),
#                        'UserInfo_3': (9, 9),
#                        'UserInfo_11': (3, 3),
#                        'UserInfo_16': (6, 6),
#                        'UserInfo_1': (9, 9),
#                        'UserInfo_6': (3, 3),
#                        'UserInfo_23': (31, 15),
#                        'UserInfo_9': (4, 4),
#                        'Education_Info3': (3, 3),
#                        'Education_Info5': (2, 2),
#                        'Education_Info1': (2, 2),
#                        'Education_Info7': (2, 2),
#                        'Education_Info6': (6, 6),
#                        'Education_Info8': (7, 7),
#                        'Education_Info2': (7, 7),
#                        'Education_Info4': (6, 6),
#                        'SocialNetwork_2': (3, 3),
#                        'SocialNetwork_7': (3, 3),
#                        'SocialNetwork_12': (3, 3),
#                        'WeblogInfo_19': (8, 8),
#                        'WeblogInfo_21': (5, 5),
#                        'WeblogInfo_20': (45, 15)}

# embedding_shape_map = {
#     'UserInfo_13': (3, 3),
#     'UserInfo_12': (3, 3),
#     'UserInfo_22': (9, 9),
#     'UserInfo_17': (2, 2),
#     'UserInfo_21': (2, 2),
#     'UserInfo_5': (3, 3),
#     'UserInfo_3': (9, 9),
#     'UserInfo_11': (3, 3),
#     'UserInfo_16': (6, 6),
#     'UserInfo_1': (9, 9),
#     'UserInfo_6': (3, 3),
#     'UserInfo_23': (31, 15),
#     'UserInfo_9': (4, 4),
#     'Education_Info3': (3, 2),
#     'Education_Info5': (2, 1),
#     'Education_Info1': (2, 1),
#     'Education_Info7': (2, 1),
#     'Education_Info6': (6, 5),
#     'Education_Info8': (7, 6),
#     'Education_Info2': (7, 6),
#     'Education_Info4': (6, 5),
#     'SocialNetwork_2': (3, 2),
#     'SocialNetwork_7': (3, 2),
#     'SocialNetwork_12': (3, 2),
#     'WeblogInfo_19': (8, 6),
#     'WeblogInfo_21': (5, 4),
#     'WeblogInfo_20': (45, 15)}
