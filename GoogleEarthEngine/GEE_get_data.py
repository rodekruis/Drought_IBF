import ee
from GEE_utils import extract_data_EE, fcdict_to_df

# initialize GEE
ee.Initialize()

# define dictionary of countries, with location of admin level shapefiles on GEE
country_dict = {
    "KE": 'users/jacopomargutti/ken_admbnda_adm1',
    "UG": 'users/jacopomargutti/uga_admbnda_adm1'
}

# loop over countries
for country_abb, admin_file in country_dict.items():

    # get country admin level shapefile
    country = ee.FeatureCollection(admin_file)
    # year range
    year_start, year_end = 1981, 2019

    print("-----------------------------------------------------------------------------------------")
    print("                                 NDVI mean                                               ")
    print("-----------------------------------------------------------------------------------------")

    # Collect Features:
    fc_ndvi_mean = extract_data_EE(im_col="NASA/GIMMS/3GV0",
                                   fe_col=country,
                                   min_year=year_start,
                                   max_year=year_end,
                                   min_month=1,
                                   max_month=12,
                                   reducer_time=ee.Reducer.mean(),
                                   reducer_space=ee.Reducer.mean())

    # Turn feature collection dict to a single dataframe
    df_ndvi_mean = fcdict_to_df(year_start, fc_ndvi_mean)
    df_ndvi_mean = df_ndvi_mean.rename(columns={'mean': 'ndvi_mean'})
    df_ndvi_mean.to_csv(country_abb + "_ndvi_adm1.csv", sep=',', index=False)

    print("-----------------------------------------------------------------------------------------")
    print("                                 CHIRPS sum (monthly cumulative)                         ")
    print("-----------------------------------------------------------------------------------------")

    # Collect Features:
    fc_p_sum = extract_data_EE(im_col="UCSB-CHG/CHIRPS/DAILY",
                               fe_col=country,
                               min_year=year_start,
                               max_year=year_end,
                               min_month=1,
                               max_month=12,
                               reducer_time=ee.Reducer.sum(),
                               reducer_space=ee.Reducer.mean())

    # Turn feature collection dict to a single dataframe
    df_p_sum = fcdict_to_df(year_start, fc_p_sum)
    df_p_sum = df_p_sum.rename(columns={'sum': 'precipitation'})
    df_p_sum.to_csv(country_abb + "_p_adm1.csv", sep=',', index=False)

    print("-----------------------------------------------------------------------------------------")
    print("                                 Land Surface Temperature mean                           ")
    print("-----------------------------------------------------------------------------------------")

    # Collect Features:
    fc_lst_mean = extract_data_EE(im_col="MODIS/006/MOD11A1",
                                  fe_col=country,
                                  min_year=year_start,
                                  max_year=year_end,
                                  min_month=1,
                                  max_month=12,
                                  reducer_time=ee.Reducer.mean(),
                                  reducer_space=ee.Reducer.mean())

    # Turn feature collection dict to a single dataframe
    df_lst_mean = fcdict_to_df(year_start, fc_lst_mean)
    df_lst_mean = df_lst_mean.rename(columns={'mean': 'lst_mean'})
    df_lst_mean.to_csv(country_abb + "_lst_adm1.csv", sep=',', index=False)

    print("-----------------------------------------------------------------------------------------")
    print("                                 soil moisture mean                                      ")
    print("-----------------------------------------------------------------------------------------")

    # Collect Features:
    fc_soilmois_mean = extract_data_EE(im_col="NASA_USDA/HSL/soil_moisture",
                                       fe_col=country,
                                       min_year=year_start,
                                       max_year=year_end,
                                       min_month=1,
                                       max_month=12,
                                       reducer_time=ee.Reducer.mean(),
                                       reducer_space=ee.Reducer.mean())

    # Turn feature collection dict to a single dataframe
    df_soilmois_mean = fcdict_to_df(year_start, fc_soilmois_mean)
    df_soilmois_mean = df_soilmois_mean.rename(columns={'mean': 'soilmois_mean'})
    # save NDVI dataframe to .CSV
    df_soilmois_mean.to_csv(country_abb + "_sm_adm1.csv", sep=',', index=False)






