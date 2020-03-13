'''
This file contains a couple of functions to convert the GIS data into datasets /time series

'''

import numpy as np
import pandas as pd

# --- these two packages are used to work with the GIS (Geographic Information System) data ---
import fiona
import rasterio
import rasterio.mask


'''
main functions to call 
'''
def extract_zonal_stats(file_shp, file_tiff, name_property, pcode_property,
                        minval=-np.inf,
                        maxval=np.inf):
    '''
    This function will open the shapefile and image file and perform 'zonal statistics'.
    'rasterio' has built-in 'mask' options, which we use to calculate the aggregrated data values per mask

    results are stored into a dataframe with the features per shape/boundary. Features stored are:
    'mean','min' and 'max'

    (rasterstats package also has a 'zonal_stats' function, decided not to use, was a bit slower it seemed)
    '''
    # --- initialize ----
    avg_vals = []
    max_vals = []
    min_vals = []
    med_vals = []

    df_out = pd.DataFrame()

    # --- open the shape file and acces wanted info ----
    shapes, names, pcode = unpack_shp(file_shp,name_property, pcode_property)

    # --- use rasterio to read the satelite image (raster data/vector image) ----
    with rasterio.open(file_tiff, 'r') as src:
        # --- acces the image (the matrix with intensity values) ----
        img = src.read(1)
        # --- perform the mask operation for every shape in the shapefile ---
        for shape in shapes:
            out_image, out_transform = rasterio.mask.mask(src, [shape], crop=True)

            # --- show the masked image (shows non-zero value within boundaries, zero outside of boundaries shape) ----
            img = out_image[0, :, :]

            # --- filter out those pixels that fall outside the boundaries of the clipping mask
            # ---- (these are set to a value indicated in the rasterfile as 'nodatavals') ----

            # --- for DMP: physical min is zero ----
            data = img[(img >= minval) & (img <= maxval)]

            # --- determine metrics ---
            avg_vals.append(data.mean())
            min_vals.append(data.min())
            max_vals.append(data.max())
            med_vals.append(np.median(data))

    # --- store results into dataframe ---
    df_out['pcode'] = pcode
    df_out['area'] = names
    df_out['mean'] = avg_vals
    df_out['min'] = min_vals
    df_out['max'] = max_vals
    df_out['med'] = med_vals
    return df_out, data, img




'''
utility functions 
'''
def unpack_shp(file_shp, name_property='WOREDANAME', pcode_property='WOR_P_CODE'):
    '''
    Get the geometries + usefull metadata from shapefile (.shp)

    :param file_shp:
    :param name_property:
    :param pcode_property:
    :return:
    '''
    # --- use fiona to read the shape file ---
    with fiona.open(file_shp, 'r') as shapefile:
        # --- store all the shapes stored in this file ---
        shapes = [feature["geometry"] for feature in shapefile]

        # --- store the boundary names ---
        names = [feature['properties'][name_property] for feature in shapefile]
        pcode = [feature['properties'][pcode_property] for feature in shapefile]

    return shapes, names, pcode







