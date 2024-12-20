"""
Author: Charles Song
Date: 12/2024
Python Version: 3.11
"""

import os
from osgeo import gdal, osr, ogr
import numpy as np
from scipy.ndimage import gaussian_filter, label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import dilation, disk
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm


def run_segmentation(chm_name,
                     data_dir="data", output_dir="output", 
                     smoothing_sigma=1.5,
                     min_distance = 2,
                     compactness=0, 
                     min_tree_height=3, min_crown_area=20, min_circularity=0,
                     ):
    """Execute the tree segmentation workflow.
    1. Loads and preprocesses the CHM data
    2. Applies Gaussian smoothing
    3. Detects local maxima (tree tops)
    4. Performs watershed segmentation
    5. Filters out non-tree structure segments
    6. Saves results as raster and vector files

    Parameters
    ----------
    min_distance : int, optional
        The minimal allowed distance separating peaks in function `peak_local_max()`.
    """
    # Load and process CHM
    chm_file = os.path.join(data_dir, chm_name)
    basename = os.path.basename(chm_file)[0:-4]
    chm_array, chm_array_metadata = raster2array(chm_file)
    chm_array[chm_array < min_tree_height] = 0
    print("Canopy height model loaded... \n")

    chm_array_smooth = gaussian_filter(chm_array, smoothing_sigma, mode='constant', truncate=2.0)
    chm_array_smooth[chm_array == 0] = 0

    array2raster(chm_array_smooth, os.path.join(output_dir, basename + '_smoothed.tif'), chm_array_metadata)
    print("Smoothed CHM saved... \n")

    local_maxi_coords = peak_local_max(chm_array_smooth, min_distance=min_distance, threshold_abs=min_tree_height, exclude_border=False)
    print("Tree tops detected... \n")
    
    local_maxi_mask = np.zeros_like(chm_array_smooth, dtype=int)
    local_maxi_mask[tuple(local_maxi_coords.T)] = 1
    markers, _ = label(local_maxi_mask)
    chm_mask = chm_array_smooth > 0

    chm_labels = watershed(chm_array_smooth, markers, mask=chm_mask, compactness=compactness)

    array2raster(chm_labels, os.path.join(output_dir, basename + '_labels.tif'), chm_array_metadata, GDALDataType="int")
    print("Segmentation Done... \n")
    
    filtered_labels = filter_segments(chm_labels, chm_array_smooth, min_crown_area, min_circularity)

    array2raster(filtered_labels, os.path.join(output_dir, basename + '_labels_filtered.tif'), chm_array_metadata, GDALDataType="int")
    print("Segments filtered and saved as a raster file... \n")

    tree_tops = local_maxima_to_points(local_maxi_coords, chm_array_smooth, filtered_labels, chm_array_metadata)
    tree_tops.to_file(os.path.join(output_dir, basename + '_tree_tops.shp'))
    print("Tree top saved. \n")

    raster_to_polygons(os.path.join(output_dir, basename + '_labels_filtered.tif'), 
                       os.path.join(output_dir, basename + '_segmentations.shp'))
    print("Segments saved as a shapefile... \n")

    print("Segmentation Complete.")


def raster2array(geotif_file):
    """Convert a GeoTIFF raster to a numpy array with associated metadata.

    Parameters
    ----------
    geotif_file : str
        Path to the input GeoTIFF file.

    Returns
    -------
    tuple
        - numpy.ndarray : The raster data as a 2D numpy array
        - dict : Metadata dictionary containing:
            - array_rows : Number of rows in the raster
            - array_cols : Number of columns in the raster
            - driver : GDAL driver name
            - projection : Spatial reference system
            - epsg : EPSG code of the spatial reference system
            - geotransform : GDAL geotransform tuple
            - pixelWidth : Width of a pixel in map units
            - pixelHeight : Height of a pixel in map units
            - ext_dict : Dictionary of extent coordinates
            - extent : Tuple of extent coordinates
            - noDataValue : NoData value in the raster
            - scaleFactor : Scale factor for pixel values
            - bandstats : Dictionary of band statistics

    Raises
    ------
    ValueError
        If the input raster has more than one band.
    """
    if not os.path.exists(geotif_file):
        raise FileNotFoundError(f"The file '{geotif_file}' does not exist.")
    dataset = gdal.Open(geotif_file)
    if dataset.RasterCount != 1:
        raise ValueError('Function only supports single band data')

    metadata = {}
    metadata['RasterYSize'], metadata['RasterXSize'] = dataset.RasterYSize, dataset.RasterXSize
    metadata['driver'] = dataset.GetDriver().LongName
    metadata['projection'] = dataset.GetProjection()
    metadata['epsg'] = int(osr.SpatialReference(wkt=dataset.GetProjection()).GetAttrValue('AUTHORITY',1))
    metadata['geotransform'] = dataset.GetGeoTransform()
    metadata['pixelWidth'], metadata['pixelHeight'] = metadata['geotransform'][1], metadata['geotransform'][5]

    metadata["extent_dict"] = {
        "xMin": metadata['geotransform'][0],
        "xMax": metadata['geotransform'][0] + dataset.RasterXSize * metadata['geotransform'][1],
        "yMin": metadata['geotransform'][3] + dataset.RasterYSize * metadata['geotransform'][5],
        "yMax": metadata['geotransform'][3],
        }

    metadata["extent"] = (
        metadata["extent_dict"]["xMin"],
        metadata["extent_dict"]["xMax"],
        metadata["extent_dict"]["yMin"],
        metadata["extent_dict"]["yMax"],
        )

    raster = dataset.GetRasterBand(1)
    if not raster.GetNoDataValue():
        raise ValueError('Raster NoDataValue should be set.')
    metadata['noDataValue'] = raster.GetNoDataValue()
    metadata['scaleFactor'] = raster.GetScale()

    metadata['bandstats'] = {}
    stats = raster.GetStatistics(True, True)
    metadata["bandstats"]["min"] = round(stats[0], 2)
    metadata["bandstats"]["max"] = round(stats[1], 2)
    metadata["bandstats"]["mean"] = round(stats[2], 2)
    metadata["bandstats"]["stdev"] = round(stats[3], 2)

    array = raster.ReadAsArray().astype(np.float32)
    array[array == metadata["noDataValue"]] = 0
    if metadata["scaleFactor"]:
        array = array / metadata["scaleFactor"]
    return array, metadata


def array2raster(array, file_path, metadata, GDALDataType="float"):
    """Save a numpy array as a GeoTIFF raster file.

    Parameters
    ----------
    array : numpy.ndarray
        2D array containing the raster data.
    file_path : str
        Path to the output GeoTIFF file.
    metadata : dict
        Metadata dictionary from raster2array function containing geotransform and projection information.

    Returns
    -------
    None
        Creates a GeoTIFF file at the specified location.
    """
    cols, rows = array.shape[1], array.shape[0]

    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    driver = gdal.GetDriverByName('GTiff')
    if GDALDataType == "float":
        eType = gdal.GDT_Float32
    elif GDALDataType == "int":
        eType = gdal.GDT_Int16
    outRaster = driver.Create(file_path, cols, rows, 1, eType)
    outRaster.SetGeoTransform((metadata['extent_dict']['xMin'], metadata['pixelWidth'], 0, 
                               metadata['extent_dict']['yMax'], 0, metadata['pixelHeight']))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outband.SetNoDataValue(-9999)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(metadata['epsg'])
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def filter_segments(labels, chm_array, min_crown_area=15, min_circularity=0):
    """Filter tree segments based on crown area, aspect ratio, and circularity thresholds.

    Parameters
    ----------
    labels : numpy.ndarray
        2D array of labeled segments from watershed segmentation.
    chm_array : numpy.ndarray
        2D array of canopy height model values.
    min_crown_area : int, optional
        Minimum crown area threshold in pixels (default: 15).
    min_circularity : float, optional
        Minimum circularity to keep a segment (default: 0). 

    Returns
    -------
    numpy.ndarray
        Filtered and relabeled segments meeting the criteria.
    """
    props = regionprops(labels, intensity_image=chm_array)
    filtered_labels = np.zeros_like(labels) 

    new_label = 1
    for prop in tqdm(props, desc="Filtering segments", unit="segment"):
        crown_area = prop.area
        perimeter = prop.perimeter
        circularity = (4 * np.pi * crown_area) / (perimeter ** 2) if perimeter > 0 else 0

        if crown_area >= min_crown_area and circularity >= min_circularity:
            filtered_labels[labels == prop.label] = new_label
            new_label += 1
    
    return filtered_labels


def local_maxima_to_points(local_maxi_coords, chm_array, filtered_labels, metadata):
    """
    Filter tree tops to retain only those within filtered segments,
    and convert filtered tree tops to a GeoDataFrame with tree heights
    
    Parameters:
    -----------
    local_maxi_coords : numpy.ndarray
        Array of (row, col) coordinates of local maxima.
    chm_array : numpy.ndarray
        The CHM array containing height values.
    filtered_labels : numpy.ndarray
        2D array of filtered segments.
    metadata : dict
        Metadata dictionary from raster2array function containing geotransform and projection information.
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing:
        - geometry : Point geometries of tree locations
        - height : Tree heights from CHM
        - tree_id : Unique identifier for each tree
    """
    geotransform = metadata['geotransform']
    projection = metadata['projection']

    row_indices, col_indices = local_maxi_coords[:, 0], local_maxi_coords[:, 1]
    filtering_mask = filtered_labels[row_indices, col_indices] != 0
    filtered_tree_tops = local_maxi_coords[filtering_mask]
    
    x_coords = geotransform[0] + filtered_tree_tops[:, 1] * geotransform[1]
    y_coords = geotransform[3] + filtered_tree_tops[:, 0] * geotransform[5]
    heights = chm_array[filtered_tree_tops[:, 0], filtered_tree_tops[:, 1]]    
    
    geometries = [Point(x, y) for x, y in zip(x_coords, y_coords)]
    
    gdf = gpd.GeoDataFrame(
        {
            'height': heights,
            'tree_id': range(1, len(heights) + 1),
            'geometry': geometries
        },
        crs=projection
    )
    
    return gdf


def raster_to_polygons(raster_file, output_file, mask=None):
    """
    Converts a labeled raster array into polygon features.

    Parameters
    ----------
    raster_array : numpy.ndarray
        2D array where each unique value (except 0) represents a distinct polygon label.
    output_file : str
        Path to save the output polygon shapefile.
    
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of polygons.
    """
    raster = gdal.Open(raster_file)
    band = raster.GetRasterBand(1)

    proj = raster.GetProjection()
    shp_proj = osr.SpatialReference()
    shp_proj.ImportFromWkt(proj)

    call_drive = ogr.GetDriverByName('ESRI Shapefile')
    create_shp = call_drive.CreateDataSource(output_file)
    shp_layer = create_shp.CreateLayer('layername', srs=shp_proj)

    new_field = ogr.FieldDefn(str('tree_id'), ogr.OFTInteger)
    shp_layer.CreateField(new_field)

    gdal.Polygonize(band, mask, shp_layer, 0, [], callback=None)
    
    create_shp.Destroy()
