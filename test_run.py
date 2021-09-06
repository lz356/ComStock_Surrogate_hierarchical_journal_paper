from ComStock_Surrogate import ComStock_Surrogate as ComStock_Surrogate

# building_type_list = ['full_service_restaurant',
#                         'small_office',
#                         'warehouse',
#                         'retail',
#                         'outpatient',
#                         'strip_mall',
#                         'large_office',
#                         'small_hotel',
#                         'quick_service_restaurant',
#                         'medium_office',
#                         'primary_school',
#                         'hospital',
#                         'secondary_school',
#                         'large_hotel',]

building_type_list = [
                        # 'secondary_school',
                        # 'large_office',
                        # 'hospital',
                        # 'large_hotel',
                        # 'medium_office',
                        # 'small_hotel',
                        # 'quick_service_restaurant',
                        'primary_school',
                        ]

for building_type in building_type_list:
    print(f'Processing: {building_type}')

    test_run = ComStock_Surrogate(version = 'com_reg1_02_01_01_2016',
                                  result_csv = 'data/com_reg1_02_01_01_2016/results_up00_com_reg1_02_01_01_2016_filter.parquet',
                                  raw_more_building_characteristics = 'data/com_reg1_02_01_01_2016/com_reg1_02_01_01_2016_results_filter.csv',
                                  read_data_from_S3 = False, number_of_trees = 10,
                                  building_type = building_type, validation = True,
                                  result_csv_for_prediction = 'data/com_reg1_02_01_01_2016/results_up00_com_reg1_02_01_01_2016_filter.parquet',
                                  raw_more_building_characteristics_for_prediction = 'data/com_reg1_02_01_01_2016/com_reg1_02_01_01_2016_results_filter.csv')

    test_run.whole_process_only_training()
