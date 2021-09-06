# Envrionment: UQ_qoi_bugfix
import os
import glob
import pickle
import fsspec
import pvlib
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from eulpuq.qoi.quantities_of_interest import QuantityOfInterest
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel

from Base import Base as Base

class ComStock_Surrogate():
    """
    __init__:
        version: now only tested in 'com_reg1_02_01_01_2016'
        result_csv: directory of result csv or parquet
        raw_more_building_characteristics: directory of result csv with extra building characteristics
        read_data_from_S3: True: read data from S3 and save only relevant data locally; False: read locally saved data
        number_of_trees: number of trees in the random forest algorithm
        building_type: the string of one building type
        result_csv_for_prediction: directory of result csv or parquet for prediction
        raw_more_building_characteristics_for_prediction: directory of result csv with extra building characteristics or prediction
        enduse_list: enduse of ComStock output, e.g. ['total_site_electricity_kwh']. Now only support single element list
        more_building_characteritics_list: a list of building characteristics in "raw_more_building_characteristics" to be added
        validation: whether to do the validation during training process
        root_path: the root path for running the code and for generating data and plots

    functions:
        create_folder_structure: create folder structure for data and results
        inputs_generator: generate inputs based on result csv and weather data
        output_generator: generate output from simulation results
        get_models: train or load trained machine learning models
        make_predictions: make predictions based on inputs for testing and model
        QOI_calculators: calculate QOIs based on Siby's QOI module
        whole_process_only_testing: whole process for testing
        whole_process_only_training: whole process for training
        whole_process_training_and_testingL whole process for both training and testing

    """
    # Finalize the __init___'s variables
    def __init__(self, version, result_csv, raw_more_building_characteristics, read_data_from_S3, number_of_trees,
                 building_type, more_building_characteritics_list = Base.more_building_characteritics_list,
                 root_path = os.getcwd(), validation = True, result_csv_for_prediction = 'none',
                 raw_more_building_characteristics_for_prediction = 'none',
                 enduse_list = ['total_site_electricity_kwh']):
        self.version = version
        self.result_csv = Base.process_raw_result_parquet_file_to_extract_basic_building_characteristics(result_csv)
        self.raw_more_building_characteristics = pd.read_csv(raw_more_building_characteristics).sort_values(by=['building_id'], ascending = True)
        self.read_data_from_S3 = read_data_from_S3
        self.number_of_trees = number_of_trees
        self.building_type = building_type
        self.result_csv_for_prediction = Base.process_raw_result_parquet_file_to_extract_basic_building_characteristics(result_csv_for_prediction)
        self.raw_more_building_characteristics_for_prediction = pd.read_csv(raw_more_building_characteristics_for_prediction).sort_values(by=['building_id'], ascending = True)
        self.enduse_list = enduse_list
        self.more_building_characteritics_list = more_building_characteritics_list
        self.validation = validation
        self.root_path = root_path

    def create_folder_structure(self):
        print('Creating folder structure...')
        folders = [f'data/{self.version}',f'models/{self.version}',f'plots/{self.version}',
                   f'results/{self.version}', f'temp/{self.version}', f'weather/{self.version}']
        for folder in folders:
            if not os.path.exists(os.path.join(self.root_path, folder)):
                os.makedirs(os.path.join(self.root_path, folder))

    def inputs_generator(self, train_or_test):
        print(f'Generating inputs for {train_or_test}ing...')
        if train_or_test == 'train':
            result_csv = self.result_csv
            raw_more_building_characteristics = self.raw_more_building_characteristics
        elif train_or_test == 'test':
            result_csv = self.result_csv_for_prediction
            raw_more_building_characteristics = self.raw_more_building_characteristics_for_prediction
        else:
            print("Error! Enter either 'train' or 'test' for train_or_test")

        enduse_list = self.enduse_list
        building_type = self.building_type
        more_building_characteritics_list = self.more_building_characteritics_list

        # time indicator
        building_ID_list_one_building_type = result_csv.loc[result_csv.building_type == building_type].building_id.unique()
        time_df = pd.DataFrame([])
        date_range = pd.date_range("01-01-2016 00:00", "12-30-2016 23:00", freq="60min")
        time_df['hour'] = date_range.hour
        time_df['month'] = date_range.month
        time_df['weekday'] = date_range.weekday
        time_df['weekday_indicator'] = [int(x>=5) for x in time_df['weekday']]
        time_df = pd.concat([time_df] * len(building_ID_list_one_building_type)).reset_index(drop = True)
        time_df['building_id'] = building_ID_list_one_building_type.repeat(24*365)
        time_df = time_df.reindex(columns=['building_id','hour','month','weekday','weekday_indicator'])
        # weather
        weather_df = Base.get_weather_df(weather_file_location = f'weather/{self.version}/',
                       weather_filename = 'USA_CO_Fort.Collins.Sawrs.724697_2016.epw',
                       cols = ['temp_air', 'temp_dew', 'relative_humidity', 'atmospheric_pressure',
                               'etr', 'etrn','ghi_infrared','ghi','dni','dhi','global_hor_illum',
                               'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance',
                               'wind_direction', 'wind_speed'],
                       timelag = 2
                      )
        weather_df = weather_df[0:8760]
        weather_df = pd.concat([weather_df]* len(building_ID_list_one_building_type)).reset_index(drop = True)
        # building characteristics
        building_characteristics = result_csv.loc[result_csv.building_id.isin(building_ID_list_one_building_type)]
        building_characteristics = building_characteristics.iloc[
            np.repeat(np.arange(len(building_characteristics)), 24*365)].reset_index(drop=True)
        # more building characteristics including operation status
        extracted_df = raw_more_building_characteristics[more_building_characteritics_list]
        extracted_df = extracted_df.loc[extracted_df.building_id.isin(building_ID_list_one_building_type)]
        new_extracted_df = pd.DataFrame(np.repeat(extracted_df.values,24*365,axis=0))
        new_extracted_df.columns = extracted_df.columns
        new_extracted_df = new_extracted_df.drop(columns= ['building_id'])
        new_extracted_df = new_extracted_df.astype(float)
        # operation status
        time_df_temp = pd.DataFrame([])
        date_range_temp = pd.date_range("01-01-2016 00:00", "12-30-2016 23:00", freq="1440min")
        time_df_temp['weekday'] = date_range_temp.weekday
        time_df_temp['weekday_indicator'] = [int(x>=5) for x in time_df_temp['weekday']]
        time_df_indicator = time_df_temp.weekday_indicator
        full_schedule_list = []
        for building_id in building_ID_list_one_building_type:
            temp = result_csv.loc[result_csv.building_id == building_id]
            weekday_start_time = temp['build_existing_model.create_typical_building_from_model_wkdy_op_hrs_start_time'].values[0]
            weekday_duration_time = temp ['build_existing_model.create_typical_building_from_model_wkdy_op_hrs_duration'].values[0]
            weekend_start_time = temp ['build_existing_model.create_typical_building_from_model_wknd_op_hrs_start_time'].values[0]
            weekend_duration_time = temp ['build_existing_model.create_typical_building_from_model_wknd_op_hrs_duration'].values[0]
            weekday_stop_time = weekday_duration_time + weekday_start_time
            weekend_stop_time = weekend_duration_time + weekend_start_time
            for x in time_df_indicator:
                if x == 0:
                    full_schedule_list += [int(x>=weekday_start_time and x<=weekday_stop_time) for x in range(24)]
                else:
                    full_schedule_list += [int(x>=weekend_start_time and x<=weekend_stop_time) for x in range(24)]
        new_extracted_df['operation_status'] = full_schedule_list

        inputs = pd.concat([building_characteristics, weather_df, time_df, new_extracted_df], axis = 1)
        # drop duplicated columns: building_id
        inputs = inputs.loc[:,~inputs.columns.duplicated()]#.iloc[:,2:]
        inputs = dd.from_pandas(inputs, npartitions=10)

        if train_or_test == 'train':
            self.inputs_train = inputs
        elif train_or_test == 'test':
            self.inputs_test = inputs
        else:
            print("Error! Enter either 'train' or 'test' for train_or_test")

    def output_generator(self):
        print('Generating output...')
        read_data_from_S3 = self.read_data_from_S3
        version = self.version
        result_csv = self.result_csv
        enduse_list = self.enduse_list
        building_type = self.building_type

        if read_data_from_S3:
            Base.get_ComStock_simulation_data_from_S3(version, enduse_list)
        else:
            # Read preread data
            df_raw = pd.read_csv(f'data/{version}/df_raw.csv')
        building_ID_list_one_building_type = result_csv.loc[result_csv.building_type == building_type].building_id.unique()
        df_raw = df_raw.loc[df_raw.building_id.isin(building_ID_list_one_building_type)].reset_index(drop = True)

        self.output_train_annual = dd.from_pandas(df_raw[enduse_list], npartitions=10)

        self.output_train_annual_total = df_raw.groupby(df_raw.index // 8760).sum()
        #self.output_train_annual_total = self.output_train_annual_total.iloc[np.repeat(np.arange(len(self.output_train_annual_total)), 8760)].reset_index(drop = True)
        self.output_train_annual_total = dd.from_pandas(self.output_train_annual_total[enduse_list], npartitions=10)

        #df_raw[enduse_list[0]] = df_raw[enduse_list[0]] / (self.inputs_train['build_existing_model.create_bar_from_building_type_ratios_total_bldg_floor_area']).compute()
        #df_raw[enduse_list[0]] = df_raw[enduse_list[0]] / (self.inputs_train['build_existing_model.rentable_area']).compute()
        temp_annual_energy = df_raw.groupby(df_raw.index // 8760).sum()
        temp_annual_energy = temp_annual_energy.iloc[np.repeat(np.arange(len(temp_annual_energy)), 8760)].reset_index(drop = True)
        df_raw[enduse_list[0]] = df_raw[enduse_list[0]] / temp_annual_energy[enduse_list[0]]

        # output_shape = dd.from_pandas(df_raw[enduse_list], npartitions=10)
        # self.output_train_shape = output_shape

        self.output_train = dd.from_pandas(df_raw[enduse_list], npartitions=10)

    def bad_building_classifier(self, train_or_load_model):
        # version = self.version
        # result_csv = self.result_csv
        # building_type = self.building_type
        pass


    def get_models(self, train_or_load_model):
        version = self.version
        result_csv = self.result_csv
        building_type = self.building_type
        if train_or_load_model == 'train':
            print(f'Training Model: {building_type} ...')
            # Here we train the annual building energy consumption model
            annual_load_model = RandomForestRegressor(n_estimators = 20, random_state=42)
            annual_load_model.fit(self.inputs_train.iloc[:,2:].groupby(self.inputs_train.iloc[:,2:].index //8760).mean(), self.output_train_annual_total.iloc[:,0].values.ravel())

            feature_importance_df_annual_energy = pd.DataFrame([])
            feature_importance_df_annual_energy['feature_name'] = self.inputs_train.iloc[:,2:].columns
            feature_importance_df_annual_energy['feature_importance'] = annual_load_model.feature_importances_
            importance_mean = np.quantile(feature_importance_df_annual_energy.feature_importance, 0.5)
            feature_importance_df_annual_energy = feature_importance_df_annual_energy.loc[feature_importance_df_annual_energy.feature_importance >= importance_mean].iloc[:,0].tolist()

            self.feature_importance_df_annual_energy = feature_importance_df_annual_energy

            annual_load_model = RandomForestRegressor(n_estimators = 20, random_state=42)
            annual_load_model.fit(self.inputs_train[feature_importance_df_annual_energy].groupby(self.inputs_train[feature_importance_df_annual_energy].index //8760).mean(), self.output_train_annual_total.iloc[:,0].values.ravel())

            # # Here we do a cross validation to decide the optimal features
            kf = KFold(n_splits=5)
            building_ID_list_one_building_type = result_csv.loc[result_csv.building_type == building_type].building_id.unique()
            selected_feature_list = []

            for train_index, test_index in kf.split(building_ID_list_one_building_type):
                building_id_list_for_training = [building_ID_list_one_building_type[index] for index in train_index]

                X_train = (self.inputs_train).loc[(self.inputs_train).building_id.isin(building_id_list_for_training)].iloc[:,2:]
                y_train = (self.output_train).loc[(self.inputs_train).building_id.isin(building_id_list_for_training)]

                regr = RandomForestRegressor(n_estimators = 5, random_state=42)
                regr.fit(X_train, y_train.iloc[:,0].values.ravel())
                feature_importance_df = pd.DataFrame([])
                feature_importance_df['feature_name'] = X_train.columns
                feature_importance_df['feature_importance'] = regr.feature_importances_
                importance_mean = np.quantile(feature_importance_df.feature_importance, 0.75)
                feature_importance_df = feature_importance_df.loc[feature_importance_df.feature_importance >= importance_mean]

                feature_importance_df_final = pd.DataFrame([])
                feature_importance_df_final[self.building_type] = feature_importance_df.sort_values(
                    ['feature_importance'], ascending = False).feature_name.values
                temp_feature_list = feature_importance_df_final[self.building_type].tolist()
                selected_feature_list += temp_feature_list

            selected_feature_list = list(set(selected_feature_list))
            self.selected_feature_list = selected_feature_list

            # Here we did the random split of building ID for training and testing
            #building_ID_list_one_building_type = result_csv.loc[result_csv.building_type == building_type].building_id.unique()
            train_building_id, test_building_id = train_test_split(building_ID_list_one_building_type,test_size=0.20, random_state=42)

            # X_train = (self.inputs_train).loc[(self.inputs_train).building_id.isin(train_building_id)].iloc[:,2:]
            # X_test = (self.inputs_train).loc[(self.inputs_train).building_id.isin(test_building_id)].iloc[:,2:]
            X_train = (self.inputs_train).loc[(self.inputs_train).building_id.isin(train_building_id)][selected_feature_list]
            X_test = (self.inputs_train).loc[(self.inputs_train).building_id.isin(test_building_id)][selected_feature_list]

            y_train = (self.output_train).loc[(self.inputs_train).building_id.isin(train_building_id)]
            y_test = (self.output_train).loc[(self.inputs_train).building_id.isin(test_building_id)]

            y_train_annual = (self.output_train_annual).loc[(self.inputs_train).building_id.isin(train_building_id)]
            y_test_annual = (self.output_train_annual).loc[(self.inputs_train).building_id.isin(test_building_id)]

            regr = RandomForestRegressor(n_estimators = self.number_of_trees, random_state=42) #max_depth = 10,
            # regr = Earth()
            regr.fit(X_train, y_train.iloc[:,0].values.ravel())
            self.model = regr
            pickle.dump(self.model, open(f'models/{version}/{building_type}.sav', 'wb'))
            print('Training Model Completed!')

            if self.validation == True:
                feature_importance_df_final = pd.DataFrame([])
                error_df = pd.DataFrame([])

                feature_importance_df = pd.DataFrame([])
                feature_importance_df['feature_name'] = X_train.columns
                feature_importance_df['feature_importance'] = regr.feature_importances_

                feature_importance_df_final [self.building_type] = feature_importance_df.sort_values(
                    ['feature_importance'], ascending = False).feature_name.values

                # y_train_predicted = regr.predict(X_train)
                # y_test_predicted = regr.predict(X_test)
                temp_train = (self.inputs_train).loc[(self.inputs_train).building_id.isin(train_building_id)][feature_importance_df_annual_energy]
                temp_test = (self.inputs_train).loc[(self.inputs_train).building_id.isin(test_building_id)][feature_importance_df_annual_energy]

                temp_2_train = pd.DataFrame(annual_load_model.predict(temp_train.groupby(temp_train.index //8760).mean()))
                temp_2_test = pd.DataFrame(annual_load_model.predict(temp_test.groupby(temp_test.index //8760).mean()))

                # print(temp_2_train.iloc[np.repeat(np.arange(len(temp_2_train)), 8760)].reset_index(drop = True))
                # print(len(temp_2_train.iloc[np.repeat(np.arange(len(temp_2_train)), 8760)].reset_index(drop = True)))
                #
                # print(regr.predict(X_train))
                #
                # print(type(regr.predict(X_train)))
                # print(len(regr.predict(X_train)))
                # print(regr.predict(X_train).shape)
                #
                # print(pd.Series(regr.predict(X_train)))
                y_train_predicted = pd.Series(regr.predict(X_train)) * temp_2_train.iloc[np.repeat(np.arange(len(temp_2_train)), 8760)].reset_index(drop = True).iloc[:,0]
                y_test_predicted = pd.Series(regr.predict(X_test)) * temp_2_test.iloc[np.repeat(np.arange(len(temp_2_test)), 8760)].reset_index(drop = True).iloc[:,0]
                self.y_test_predicted = y_test_predicted

                # y_train = y_train.compute()
                # y_train_predicted = y_train_predicted
                # y_test = y_test.compute()
                # y_test_predicted = y_test_predicted
                y_train = y_train_annual.compute()
                #y_train_predicted = y_train_predicted
                y_test = y_test_annual.compute()
                self.y_test = y_test
                #y_test_predicted = y_test_predicted

                training_error = (mean_squared_error(y_train.values, y_train_predicted))**0.5/y_train.mean()
                testing_error = (mean_squared_error(y_test.values, y_test_predicted))**0.5/y_test.mean()

                saved_result_df = pd.concat([pd.DataFrame(y_train.values), pd.DataFrame(y_train_predicted),
                                             pd.DataFrame(y_test.values), pd.DataFrame(y_test_predicted)], axis = 1)
                saved_result_df.columns = ['y_train','_y_train_predicted','y_test','y_test_predicted']
                saved_result_df.to_csv(f'results/{self.version}/y_train_and_y_test_{self.building_type}.csv', index = None)

                error_df[building_type] = [training_error.values[0], testing_error.values[0]]
                error_df.index = ['training_error','testing_error']
                error_df.to_csv(f'results/{self.version}/error_{self.building_type}.csv')

                feature_importance_df_final.to_csv(f'results/{self.version}/feature_importance_{self.building_type}.csv')

                # plots module is added here

        elif train_or_load_model == 'load':
            print(f'Loading Model: {building_type} ...')
            self.model = pickle.load(open(f'models/{version}/{building_type}.sav', 'rb'))
            print('Loading Model Completed!')
        else:
            print ("Error! Enter either 'train' or 'load' for train_or_load_model")
    # to be worked on this def
    def make_predictions(self):
        version = self.version
        building_type = self.building_type
        self.output_test = self.model.predict(self.inputs_test.iloc[:,2:])
        pd.DataFrame(self.output_test, columns = ['output_test']).to_csv(f'results/{version}/predictions_{building_type}.csv', index = None)
        print('Make and saving predictions...')

    def QOI_calculators(self):
        print('Calculating and saving QOIs...')
        temp_dask_df = pd.DataFrame(self.y_test_predicted, columns = ['total_site_electricity_kwh'])
        temp_dask_df ['Geometry_Building_Type_RECS'] = self.building_type
        temp_dask_df ['time'] = np.repeat(pd.date_range("01-01-2016 00:00", "12-30-2016 23:00", freq="60min"), int(len(temp_dask_df)/(365*24)))
        temp_dask_df ['Building'] = 0
        temp_dask_df = dd.from_pandas(pd.DataFrame(temp_dask_df), npartitions = 10)

        QOI_model = QuantityOfInterest(temp_dask_df)
        QOI_model.update_dask_df(temp_dask_df)
        temp_dict = QOI_model.get_all_qoi(building_type=self.building_type)
        self.QOI_calculation_results = pd.DataFrame.from_dict(temp_dict, orient='index', columns = ['total_site_electricity_kwh_or_time_of_day'])
        self.QOI_calculation_results.to_csv(f"results/{self.version}/QOI_calculations_{self.building_type}_validation_predict.csv")

        print('Calculating and saving QOIs...')
        temp_dask_df = pd.DataFrame(self.y_test, columns = ['total_site_electricity_kwh'])
        temp_dask_df ['Geometry_Building_Type_RECS'] = self.building_type
        temp_dask_df ['time'] = np.repeat(pd.date_range("01-01-2016 00:00", "12-30-2016 23:00", freq="60min"), int(len(temp_dask_df)/(365*24)))
        temp_dask_df ['Building'] = 0
        temp_dask_df = dd.from_pandas(pd.DataFrame(temp_dask_df), npartitions = 10)

        QOI_model = QuantityOfInterest(temp_dask_df)
        QOI_model.update_dask_df(temp_dask_df)
        temp_dict = QOI_model.get_all_qoi(building_type=self.building_type)
        self.QOI_calculation_results = pd.DataFrame.from_dict(temp_dict, orient='index', columns = ['total_site_electricity_kwh_or_time_of_day'])
        self.QOI_calculation_results.to_csv(f"results/{self.version}/QOI_calculations_{self.building_type}_validation_truth.csv")

    def whole_process_only_training(self):
        self.create_folder_structure()
        self.inputs_generator(train_or_test = 'train')
        self.output_generator()
        self.get_models(train_or_load_model = 'train')
        self.QOI_calculators()

    def whole_process_training_and_testing(self):
        self.create_folder_structure()
        self.inputs_generator(train_or_test = 'train')
        self.output_generator()
        self.get_models(train_or_load_model = 'train')
        self.inputs_generator(train_or_test = 'test')
        self.make_predictions()
        self.QOI_calculators()

    def whole_process_only_testing(self):
        self.create_folder_structure()
        self.get_models(train_or_load_model = 'load')
        self.inputs_generator(train_or_test = 'test')
        self.make_predictions()
        self.QOI_calculators()
