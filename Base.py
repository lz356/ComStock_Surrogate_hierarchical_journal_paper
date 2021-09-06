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

class Base():
    """
    The class "Base" provides constants and base functions to be used in class "ComStock_Surrogate".
    """
    more_building_characteritics_list = ['building_id', # building_id has to be in the first element of this list
                                       #'com_stock_sensitivity_reports.applicable',
                                       'com_stock_sensitivity_reports.com_report_air_system_fan_power_minimum_flow_fraction',
                                       # 'com_stock_sensitivity_reports.com_report_air_system_fan_static_pressure',
                                       'com_stock_sensitivity_reports.com_report_air_system_fan_total_efficiency',
                                       #'com_stock_sensitivity_reports.com_report_average_boiler_efficiency',
                                       #'com_stock_sensitivity_reports.com_report_average_chiller_cop',
                                       #'com_stock_sensitivity_reports.com_report_average_cooling_setpoint_max_c',
                                       #'com_stock_sensitivity_reports.com_report_average_cooling_setpoint_min_c',
                                       #'com_stock_sensitivity_reports.com_report_average_dx_cooling_cop',
                                       #'com_stock_sensitivity_reports.com_report_average_dx_heating_cop',
                                       #'com_stock_sensitivity_reports.com_report_average_exterior_wall_u_value_si',
                                       #'com_stock_sensitivity_reports.com_report_average_gas_coil_efficiency',
                                       #'com_stock_sensitivity_reports.com_report_average_heating_setpoint_max_c',
                                       #'com_stock_sensitivity_reports.com_report_average_heating_setpoint_min_c',
                                       #'com_stock_sensitivity_reports.com_report_average_outdoor_air_fraction',
                                       #'com_stock_sensitivity_reports.com_report_average_roof_absorptance',
                                       #'com_stock_sensitivity_reports.com_report_average_roof_u_value_si',
                                       'com_stock_sensitivity_reports.com_report_building_fraction_cooled',
                                       'com_stock_sensitivity_reports.com_report_building_fraction_heated',
                                       'com_stock_sensitivity_reports.com_report_daylight_control_fraction',
                                       'com_stock_sensitivity_reports.com_report_design_chiller_cop',
                                       'com_stock_sensitivity_reports.com_report_design_dx_cooling_cop',
                                       'com_stock_sensitivity_reports.com_report_design_dx_heating_cop',
                                       'com_stock_sensitivity_reports.com_report_design_outdoor_air_flow_rate_m_3_per_m_2_s',
                                       #'com_stock_sensitivity_reports.com_report_elevator_energy_use_gj',
                                       #'com_stock_sensitivity_reports.com_report_exterior_lighting_power_w',
                                       #'com_stock_sensitivity_reports.com_report_hot_water_volume_m_3',
                                       'com_stock_sensitivity_reports.com_report_interior_electric_equipment_eflh',
                                       'com_stock_sensitivity_reports.com_report_interior_electric_equipment_power_density_w_per_m_2',
                                       'com_stock_sensitivity_reports.com_report_interior_lighting_eflh',
                                       'com_stock_sensitivity_reports.com_report_interior_lighting_power_density_w_per_m_2',
                                       'com_stock_sensitivity_reports.com_report_internal_mass_area_ratio',
                                       'com_stock_sensitivity_reports.com_report_occupant_density_ppl_per_m_2',
                                       'com_stock_sensitivity_reports.com_report_occupant_eflh',
                                       'com_stock_sensitivity_reports.com_report_window_shgc',
                                       'com_stock_sensitivity_reports.com_report_window_u_value_si',
                                       'com_stock_sensitivity_reports.com_report_wwr',
                                       'com_stock_sensitivity_reports.com_report_zone_hvac_fan_power_minimum_flow_fraction',]
                                       #'com_stock_sensitivity_reports.com_report_zone_hvac_fan_static_pressure',
                                       #'com_stock_sensitivity_reports.com_report_zone_hvac_fan_total_efficiency']

    #more_building_characteritics_list = ['building_id']

    # more_building_characteritics_list = ['building_id', # building_id has to be in the first element of this list
    #                                    #'com_stock_sensitivity_reports.applicable',
    #                                    #'com_stock_sensitivity_reports.com_report_air_system_fan_power_minimum_flow_fraction',
    #                                    #'com_stock_sensitivity_reports.com_report_air_system_fan_static_pressure',
    #                                    #'com_stock_sensitivity_reports.com_report_air_system_fan_total_efficiency',
    #                                    #'com_stock_sensitivity_reports.com_report_average_boiler_efficiency',
    #                                    #'com_stock_sensitivity_reports.com_report_average_chiller_cop',
    #                                    #'com_stock_sensitivity_reports.com_report_average_cooling_setpoint_max_c',
    #                                    #'com_stock_sensitivity_reports.com_report_average_cooling_setpoint_min_c',
    #                                    #'com_stock_sensitivity_reports.com_report_average_dx_cooling_cop',
    #                                    #'com_stock_sensitivity_reports.com_report_average_dx_heating_cop',
    #                                    #'com_stock_sensitivity_reports.com_report_average_exterior_wall_u_value_si',
    #                                    #'com_stock_sensitivity_reports.com_report_average_gas_coil_efficiency',
    #                                    #'com_stock_sensitivity_reports.com_report_average_heating_setpoint_max_c',
    #                                    #'com_stock_sensitivity_reports.com_report_average_heating_setpoint_min_c',
    #                                    #'com_stock_sensitivity_reports.com_report_average_outdoor_air_fraction',
    #                                    #'com_stock_sensitivity_reports.com_report_average_roof_absorptance',
    #                                    #'com_stock_sensitivity_reports.com_report_average_roof_u_value_si',
    #                                    'com_stock_sensitivity_reports.com_report_building_fraction_cooled',
    #                                    'com_stock_sensitivity_reports.com_report_building_fraction_heated',
    #                                    'com_stock_sensitivity_reports.com_report_daylight_control_fraction',
    #                                    'com_stock_sensitivity_reports.com_report_design_chiller_cop',
    #                                    'com_stock_sensitivity_reports.com_report_design_dx_cooling_cop',
    #                                    'com_stock_sensitivity_reports.com_report_design_dx_heating_cop',
    #                                    'com_stock_sensitivity_reports.com_report_design_outdoor_air_flow_rate_m_3_per_m_2_s',
    #                                    #'com_stock_sensitivity_reports.com_report_elevator_energy_use_gj',
    #                                    #'com_stock_sensitivity_reports.com_report_exterior_lighting_power_w',
    #                                    #'com_stock_sensitivity_reports.com_report_hot_water_volume_m_3',
    #                                    #'com_stock_sensitivity_reports.com_report_interior_electric_equipment_eflh',
    #                                    #'com_stock_sensitivity_reports.com_report_interior_electric_equipment_power_density_w_per_m_2',
    #                                    #'com_stock_sensitivity_reports.com_report_interior_lighting_eflh',
    #                                    #'com_stock_sensitivity_reports.com_report_interior_lighting_power_density_w_per_m_2',
    #                                    'com_stock_sensitivity_reports.com_report_internal_mass_area_ratio',
    #                                    'com_stock_sensitivity_reports.com_report_occupant_density_ppl_per_m_2',
    #                                    #'com_stock_sensitivity_reports.com_report_occupant_eflh',
    #                                    'com_stock_sensitivity_reports.com_report_window_shgc',
    #                                    'com_stock_sensitivity_reports.com_report_window_u_value_si',
    #                                    'com_stock_sensitivity_reports.com_report_wwr']
    #                                    #'com_stock_sensitivity_reports.com_report_zone_hvac_fan_power_minimum_flow_fraction',
    #                                    #'com_stock_sensitivity_reports.com_report_zone_hvac_fan_static_pressure',]
    #                                    #'com_stock_sensitivity_reports.com_report_zone_hvac_fan_total_efficiency']

    def process_raw_result_parquet_file_to_extract_basic_building_characteristics(parquet_file_directory):
        """
        This function is used to extract basic building characteristics, including
        converting TRUE and FALSE in string format to 0 and 1, converting year in
        string format to year in int, converting year of building code to year in
        int, and converting other number in string format to float or int format.

        Args:
            parquet_file_directory (str): directory of raw result parquet file
            directly downloaded from S3 bucket

        Returns:
            pandas.DataFrame: a DataFrame with building IDs and their extracted
            basic building characteristics

        Examples:
            processed_result_df = process_raw_result_parquet_file_to_extract_basic_building_characteristics(
            'data/results_up00_com_reg1_02_01_01_2016.parquet')

        """
        results_df = pd.read_parquet(parquet_file_directory)

        variables_that_are_numerical_and_directly_can_be_used = ['build_existing_model.aspect_ratio',
                                                                 'build_existing_model.create_bar_from_building_type_ratios_ns_to_ew_ratio',
                                                                 'build_existing_model.create_bar_from_building_type_ratios_num_stories_above_grade',
                                                                 'build_existing_model.create_bar_from_building_type_ratios_total_bldg_floor_area',
                                                                 'build_existing_model.mfm_tstat_code',
                                                                 'build_existing_model.number_stories',
                                                                 'build_existing_model.set_multifamily_thermostat_setpoints_tstat_index'
                                                                ]

        # remove failure runs
        results_df = results_df.loc[results_df.completed_status == 'Success'].reset_index(drop = True)
        processed_result_df = pd.DataFrame([])
        processed_result_df['building_id'] = results_df.building_id
        processed_result_df['building_type'] = results_df['build_existing_model.building_type']
        processed_result_df = pd.concat([processed_result_df, results_df[variables_that_are_numerical_and_directly_can_be_used].astype(float)], axis = 1)

        def convert_string_to_year_1 (string):
            temp = string.split('_')
            if len(string) == 8:
                return int(temp[1])
            else:
                return (int(temp[0]) + int(temp[1]))*0.5

        def convert_rotation (rotation):
            return abs(int(rotation) - 180)

        dict_1 = {
            'ComStock 90.1-2004':2004,
            'ComStock 90.1-2007':2007,
            'ComStock 90.1-2010':2010,
            'ComStock 90.1-2013':2013,
            'ComStock DOE Ref 1980-2004':1992,
            'ComStock DOE Ref Pre-1980':1980
        }

        dict_2 = {
            'true' : 1,
            'false' : 0
        }

        dict_3 = {
            'NaturalGas': 0,
            'Electricity': 1
        }

        def convert_time(time):
            if time == 'NA':
                return 0
            else:
                temp = time.split(':')
                return int(temp[0]) + int(temp[1])/60

        def convert_string_to_year_2 (string):
            temp = string.split('_')
            if temp[0] == '':
                return int(temp[1])
            elif temp[1] == '1mil':
                return int(temp[0])
            else:
                return (int(temp[0]) + int(temp[1]))*0.5

        processed_result_df['build_existing_model.built_code'] = results_df[
            'build_existing_model.built_code'].map(convert_string_to_year_1)
        processed_result_df['build_existing_model.create_bar_from_building_type_ratios_building_rotation'] = results_df[
            'build_existing_model.create_bar_from_building_type_ratios_building_rotation'].map(convert_rotation)
        processed_result_df['build_existing_model.create_bar_from_building_type_ratios_template'] = results_df[
            'build_existing_model.create_bar_from_building_type_ratios_template'].map(dict_1)
        processed_result_df['build_existing_model.create_typical_building_from_model_modify_wknd_op_hrs'] = results_df[
            'build_existing_model.create_typical_building_from_model_modify_wknd_op_hrs'].map(dict_2)
        processed_result_df['build_existing_model.create_typical_building_from_model_swh_src'] = results_df[
            'build_existing_model.create_typical_building_from_model_swh_src'].map(dict_3)
        processed_result_df['build_existing_model.create_typical_building_from_model_template'] = results_df[
            'build_existing_model.create_typical_building_from_model_template'].map(dict_1)
        processed_result_df['build_existing_model.create_typical_building_from_model_wkdy_op_hrs_duration'] = results_df[
            'build_existing_model.create_typical_building_from_model_wkdy_op_hrs_duration'].map(convert_time)
        processed_result_df['build_existing_model.create_typical_building_from_model_wkdy_op_hrs_start_time'] = results_df[
            'build_existing_model.create_typical_building_from_model_wkdy_op_hrs_start_time'].map(convert_time)
        processed_result_df['build_existing_model.create_typical_building_from_model_wknd_op_hrs_duration'] = results_df[
            'build_existing_model.create_typical_building_from_model_wknd_op_hrs_duration'].map(convert_time)
        processed_result_df['build_existing_model.create_typical_building_from_model_wknd_op_hrs_start_time'] =  results_df[
            'build_existing_model.create_typical_building_from_model_wknd_op_hrs_start_time'].map(convert_time)
        processed_result_df['build_existing_model.envelope_code'] = results_df[
            'build_existing_model.envelope_code'].map(convert_string_to_year_1)
        processed_result_df['build_existing_model.ext_lgt_code'] = results_df[
            'build_existing_model.ext_lgt_code'].map(convert_string_to_year_1)
        processed_result_df['build_existing_model.hvac_code'] = results_df[
            'build_existing_model.hvac_code'].map(convert_string_to_year_1)
        processed_result_df['build_existing_model.int_equip_code'] = results_df[
            'build_existing_model.int_equip_code'].map(convert_string_to_year_1)
        processed_result_df['build_existing_model.int_lgt_code'] = results_df[
            'build_existing_model.int_lgt_code'].map(convert_string_to_year_1)
        processed_result_df['build_existing_model.rentable_area'] = results_df[
            'build_existing_model.rentable_area'].map(convert_string_to_year_2)
        processed_result_df['build_existing_model.set_envelope_template_as_constructed_template'] = results_df[
            'build_existing_model.set_envelope_template_as_constructed_template'].map(dict_1)
        processed_result_df['build_existing_model.set_envelope_template_template'] = results_df[
            'build_existing_model.set_envelope_template_template'].map(dict_1)
        processed_result_df['build_existing_model.set_exterior_lighting_template_as_constructed_template'] = results_df[
            'build_existing_model.set_exterior_lighting_template_as_constructed_template'].map(dict_1)
        processed_result_df['build_existing_model.set_exterior_lighting_template_template'] = results_df[
            'build_existing_model.set_exterior_lighting_template_template'].map(dict_1)
        processed_result_df['build_existing_model.set_hvac_template_as_constructed_template'] = results_df[
            'build_existing_model.set_hvac_template_as_constructed_template'].map(dict_1)
        processed_result_df['build_existing_model.set_hvac_template_template'] = results_df[
            'build_existing_model.set_hvac_template_template'].map(dict_1)
        processed_result_df['build_existing_model.set_interior_equipment_template_as_constructed_template'] = results_df[
            'build_existing_model.set_interior_equipment_template_as_constructed_template'].map(dict_1)
        processed_result_df['build_existing_model.set_interior_equipment_template_template'] = results_df[
            'build_existing_model.set_interior_equipment_template_template'].map(dict_1)
        processed_result_df['build_existing_model.set_interior_lighting_template_as_constructed_template'] = results_df[
            'build_existing_model.set_interior_lighting_template_as_constructed_template'].map(dict_1)
        processed_result_df['build_existing_model.set_interior_lighting_template_template'] = results_df[
            'build_existing_model.set_interior_lighting_template_template'].map(dict_1)
        processed_result_df['build_existing_model.set_multifamily_thermostat_setpoints_template'] = results_df[
            'build_existing_model.set_multifamily_thermostat_setpoints_template'].map(dict_1)
        processed_result_df['build_existing_model.set_service_water_heating_template_as_constructed_template'] = results_df[
            'build_existing_model.set_service_water_heating_template_as_constructed_template'].map(dict_1)
        processed_result_df['build_existing_model.set_service_water_heating_template_template'] = results_df[
            'build_existing_model.set_service_water_heating_template_template'].map(dict_1)
        processed_result_df['build_existing_model.swh_code'] = results_df[
            'build_existing_model.swh_code'].map(convert_string_to_year_1)
        processed_result_df['build_existing_model.swh_src'] = results_df[
            'build_existing_model.swh_src'].map(dict_3)
        processed_result_df['build_existing_model.weekday_duration'] = results_df[
            'build_existing_model.weekday_duration'].map(convert_time)
        processed_result_df['build_existing_model.weekday_start_time'] = results_df[
            'build_existing_model.weekday_start_time'].map(convert_time)
        processed_result_df['build_existing_model.weekend_duration'] = results_df[
            'build_existing_model.weekend_duration'].map(convert_time)
        processed_result_df['build_existing_model.weekend_start_time'] = results_df[
            'build_existing_model.weekend_start_time'].map(convert_time)

        processed_result_df = processed_result_df.sort_values(by=['building_id'], ascending = True)

        return processed_result_df

    def get_weather_df(weather_file_location, weather_filename, cols=None, timelag=0, year=None,
                       start_day_of_week=None):
        """
        Get a dataframe of the weather data for an epw.
        Retrieves weather data from ``weather_file_location`` (possibly with caching)
        Create timelags of weather variables.
        For example for OAT, when timelag == 3, then generate OAT, OAT_t-1, OAT_t-2, OAT_t-3
        :param weather_file_location: folder or s3 location of weather files
        :type weather_file_location: str
        :param weather_filename: epw file to read
        :type weather_filename: str
        :param cols: column names of weather variable in epw file, defaults to all columns
        :type cols: list, optional
        :param timelag: step of timelag applied to all weather variables, defaults to 0, meaning no timelag
        :type timelag: int, optional
        :param year: year to replace the epw year(s) with. Mutually exclusive with start_day_of_week.
        :type year: int, optional
        :param start_day_of_week: Find a year that starts on this weekday 0 is Monday, 6 is Sunday.
            Mutually exclusive with year.
        :type start_day_of_week: int, optional
        """
        if not cols:
            cols = ['temp_air', 'temp_dew', 'relative_humidity', 'atmospheric_pressure',
                    'etr', 'etrn', 'ghi_infrared', 'ghi', 'dni', 'dhi', 'global_hor_illum',
                    'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance',
                    'wind_direction', 'wind_speed', 'total_sky_cover', 'opaque_sky_cover',
                    'visibility', 'ceiling_height', 'present_weather_observation',
                    'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth',
                    'snow_depth', 'days_since_last_snowfall', 'albedo',
                    'liquid_precipitation_depth', 'liquid_precipitation_quantity']

        epw_openfile = fsspec.open(f'{weather_file_location}/{weather_filename}', mode='rt')
        with epw_openfile as f:
            epw_df = pvlib.iotools.parse_epw(f, coerce_year=None)[0][cols]

        if epw_df.shape[0] == 8760:
            is_leap_year = False
        else:
            assert(epw_df.shape[0] == 8784)
            is_leap_year = True

        if year is not None:
            if start_day_of_week is not None:
                raise ValueError('Both year and start_on_weekday cannot be provided.')
            epw_df.index = epw_df.index.map(lambda x: x.replace(year=year))
        elif start_day_of_week is not None:
            assert year is None
            year = dt.date.today().year
            while True:
                if dt.date(year, 1, 1).weekday() == start_day_of_week and calendar.isleap(year) == is_leap_year:
                    break
                year -= 1
            epw_df.index = epw_df.index.map(lambda x: x.replace(year=year))

        epw_tl_df = epw_df.copy()
        for i in range(1, 1 + timelag):
            df = epw_df.shift(periods=i, freq=None, axis=0)
            epw_tl_df = epw_tl_df.join(df, how='left', rsuffix=f'_t-{i}')

        epw_tl_df = epw_tl_df.bfill(axis=0)

        return epw_tl_df

    def get_ComStock_simulation_data_from_S3(version, enduse_list):
        """
        This function is used to download ComStock simulation results from S3, do postprocessing,
        and save them to local computer. The purpose of this function is to save the downloading
        time from S3 if the data need to be donwloaded for multiple times. The data are saved to
        a temp folder f"temp/{version}/"

        Args:
            enduse_list (list of str): The list of enduses to be modeled for the surrogate
                                       or the output of the surrogate models.
            version (str): The version of ComStock run for the surrogate modeling.

        Examples:
            get_data_from_S3(enduse_list = ['total_site_electricity_kwh'], version = 'com_reg1_02_01_01_2016')

        """

        a_large_integer_exceeding_total_number_of_parquet_file_in_S3 = 10000
        for file_ID in range(a_large_integer_exceeding_total_number_of_parquet_file_in_S3):
            try:
                ddf = dd.read_parquet(f's3://eulp/simulation_output/regional_runs/comstock/{version}/{version}/timeseries/upgrade=0/part.{file_ID}.parquet',
                                              columns = ['building_id'] + enduse_list)

                df_raw = ddf.groupby(ddf.index // 4).sum().compute()
                df_raw['building_id'] = (df_raw['building_id']/4).astype(int)
                df_raw = df_raw.sort_values(by=['building_id'], ascending = True)
                df_raw.to_csv(f'temp/{version}/{file_ID}.csv', index = None)
            except:
                pass

            all_filenames = [i for i in glob.glob(f'temp/{version}/*.csv')]
            df_raw = pd.concat([pd.read_csv(f) for f in all_filenames])
            df_raw.to_csv(f'data/{version}/df_raw.csv',index = None)
