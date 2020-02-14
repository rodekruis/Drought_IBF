'''
This file contains a couple of functions to post-process the satellite data (GEE engine) and drought events



'''


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
Znorm = StandardScaler()


'''
main functions to call 
'''
def norm_by_district(DroughtData, features_no_norm):

    # --- in case this is not the first operation ---
    DroughtData.reset_index(inplace=True, drop=True)



    features = list(DroughtData.columns.drop(features_no_norm))




    normed_by_district = pd.DataFrame()
    for district in DroughtData['District'].unique():
        # -- filter by district --
        group = DroughtData[DroughtData['District'] == district].reset_index()
        satelite_data = group[features]
        Znorm.fit(satelite_data)
        normed_satelite_data = pd.DataFrame(Znorm.transform(satelite_data), columns=features)
        normed_satelite_data[features_no_norm] = group[features_no_norm]

        normed_by_district = pd.concat([normed_by_district, normed_satelite_data], ignore_index=True)


    return normed_by_district


def reduce_dataset(DroughtData, districts_with_droughts, keep_years=1):
    '''
    Drought events are recorded over short period than satellite data.
    Only keep satellite data over time period surrounding each event.
    (default is 1 year before/after each event)


    :param DroughtData:
    :param districts_with_droughts:
    :param keep_years:
    :return:
    '''


    # --- loop over all districts that have a drought recorded ----
    keepsies = []
    for district in districts_with_droughts[districts_with_droughts['drought_in_district']]['District']:
        group = DroughtData[DroughtData['District'] == district]

        # -- loop over recorded events for this district ----
        for event_index in group[group['drought_reported']].index:
            to_keep = rows_to_keep(data=DroughtData, event_index=event_index, district=district, keep_years=keep_years)
            keepsies += to_keep

    # --- only need unique set of rows (rows may be kept due to multiple drought events) ----
    keepsies = np.unique(keepsies)

    reduced_data = DroughtData.iloc[keepsies]

    print('length original data: ', len(DroughtData))
    print('length reduced data: ', len(reduced_data))
    return reduced_data


def sliding_window(DroughtData, window_size):
    '''
    Aggregrate data over a given time period into a single-valued feature.

    first version: Choice of what metrics is now hard coded. Could make variable if needed

    :param DroughtData:
    :param window_size:
    :return:
    '''
    # --- in case this is not the first operation ---
    DroughtData.reset_index(inplace=True,drop=True)


    # --- all indicators ---
    features = list(DroughtData.columns.drop(['Country', 'District', 'date', 'day', 'month', 'year',
                                              'drought_reported', 'drought_desinventar', 'drought_news_article']))


    # --- check max value in these ----
    features_min_val = ['NDVI', 'EVI', 'rainfall',
                        'precipitation_per_hour_v1',
                        'precipitation_per_hour_v2',
                        'SoilMoisture00_10cm',
                        'SoilMoisture10_40cm',
                        'SoilMoisture40_100cm',
                        'SoilMoisture100_200cm',
                        'SPEI_1month',
                        'SPEI_2month',
                        'SPEI_3month',
                        'SPEI_4month',
                        'SPEI_5month',
                        'SPEI_6month',
                        'SPEI_7month',
                        'SPEI_8month',
                        'SPEI_9month',
                        'SPEI_10month',
                        'SPEI_11month',
                        'SPEI_12month',
                        'wind_speed']

    # --- check min values in these ----
    features_max_val = ['evapotranspiration',
                        'surface_temperature_daytime',
                        'surface_temperature_nighttime',
                        'air_temperature',
                        'SoilTemperature00_10cm',
                        'SoilTemperature10_40cm',
                        'SoilTemperature40_100cm',
                        'SoilTemperature100_200cm',
                        'wind_speed']


    # --- perform the analysis ----
    sliding_window_data = pd.DataFrame()
    for district in DroughtData['District'].unique():
        group = DroughtData[DroughtData['District'] == district]

        windows = apply_sliding_window(group, window_size=window_size)

        district_data = pd.DataFrame()
        for w in windows:
            # --- slice dataframe into this window ----
            window_data = group.loc[w]
            # --- get maximum of indicators we think should positively correlate with droughts ----
            metric_max = apply_metric(window_data, columns_to_calc=features_max_val, method='max')

            # --- get minimum of indicators we think should negatively correlate with droughts ----
            metric_min = apply_metric(window_data, columns_to_calc=features_min_val, method='min')

            # --- get median values for all indicators per window -----
            metric_med = apply_metric(window_data, columns_to_calc=features, method='med')

            # --- merge together ----
            metrics_window = pd.concat([metric_max, metric_min, metric_med])

            # --- drought reported within this window? ----
            #### COULD ADD THE COLUMNS INDICATING IF IT IS DESINVENTAR OR NEWS ARTICLES ####
            metrics_window['drought_count'] = int(window_data['drought_reported'].sum())

            # --- add together ----
            district_data = pd.concat([district_data, metrics_window.to_frame().transpose()], sort=True)

        # --- add the information of the district back into dataframe ----
        district_data.reset_index(inplace=True, drop=True)
        district_data['District'] = group.reset_index()['District']
        district_data['Country'] = group.reset_index()['Country']
        district_data['Date_start_window'] = group.reset_index()['date']

        # --- merge all data (all districts) together in combined dataset -----
        sliding_window_data = pd.concat([sliding_window_data, district_data], sort=True)


    # ---  make boolean column for drought events ----
    sliding_window_data['drought_reported'] = sliding_window_data['drought_count'].astype(bool)


    return sliding_window_data



'''
utility functions 

'''

def rows_to_keep(data, event_index, district, keep_years=1):
    '''
    given the row of full dataframe containing the event, return the rows you want to keep
    '''
    event_year = data.iloc[event_index]['year']
    DistrictData = data[data['District'] == district]
    KeepData = DistrictData[(DistrictData['year'] >= (event_year - keep_years)) &
                            (DistrictData['year'] <= (event_year + keep_years))]

    return list(KeepData.index)


def apply_sliding_window(data, window_size=3):
    '''
    returns set of indices for all the overlapping windows

    at boundary of dataframe, windows are smaller
    '''
    windows = []
    for i in range(len(data)):
        start_index = max(min(data.index), min(data.index) + int(i - np.floor(0.5 * (window_size - 1))))
        stop_index = min(max(data.index), min(data.index) + int(i + np.ceil(0.5 * (window_size - 1))))
        window_index = list(range(start_index, stop_index + 1))
        windows.append(window_index)
    return windows


def apply_metric(data, columns_to_calc, method='min'):
    '''
    get the min/max/median of values within window.
    Add a column with 'feature_x' (with x in [min, max, med]) to dataframe

    :param data:
    :param columns_to_calc:
    :param method:
    :return:
    '''

    if method == 'min':
        values = data[columns_to_calc].min(skipna=True)
    elif method == 'max':
        values = data[columns_to_calc].max(skipna=True)
    elif method == 'med':
        values = data[columns_to_calc].median(skipna=True)

    change_names = {}
    for name in columns_to_calc:
        change_names[name] = name + '_' + method

    return values.rename(index=change_names)


