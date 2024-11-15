import os
from osgeo import gdal, osr
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon


def run_segmentation(chm_name,
                     data_dir="data", output_dir="output", 
                     smoothing_sigma=1.5,
                     compactness=0, 
                     min_tree_height=2.5, min_crown_area=15, min_circularity=0,
                     ):
    """Execute the tree segmentation workflow.
    1. Loads and preprocesses the CHM data
    2. Applies Gaussian smoothing
    3. Detects local maxima (tree tops)
    4. Performs watershed segmentation
    5. Filters out non-tree structure segments
    6. Saves results as raster and vector files
    """
    # Load and process CHM
    chm_file = os.path.join(data_dir, chm_name)
    if not os.path.exists(chm_file):
        raise FileNotFoundError(f"The file '{chm_file}' does not exist.")
    basename = os.path.basename(chm_file)[0:-4]
    chm_array, chm_array_metadata = raster2array(chm_file)
    chm_array[chm_array < min_tree_height] = 0

    # Smooth CHM
    chm_array_smooth = ndi.gaussian_filter(
        chm_array, smoothing_sigma, mode='constant', truncate=2.0
        )
    chm_array_smooth[chm_array == 0] = 0

    # Save smoothed CHM
    array2raster(
        np.array(chm_array_smooth, dtype=float),
        os.path.join(output_dir, basename + '_smoothed.tif'),
        chm_array_metadata
        )

    # Detect tree tops
    local_maxi_coords = peak_local_max(
        chm_array_smooth,
        # min_distance=LOCAL_MAXI_WINDOW,
        footprint=np.ones((5, 5)),
        threshold_abs=min_tree_height,
        exclude_border=False
        )
    
    local_maxi_mask = np.zeros_like(chm_array_smooth, dtype=int)
    local_maxi_mask[tuple(local_maxi_coords.T)] = 1
    markers, _ = ndi.label(local_maxi_mask)

    chm_mask = chm_array_smooth
    chm_mask[chm_array_smooth != 0] = 1

    chm_labels = watershed(chm_array_smooth, markers, mask=chm_mask, compactness=compactness)

    array2raster(
        np.array(chm_labels),
        os.path.join(output_dir, basename + '_labels.tif'),
        chm_array_metadata
        )
    
    filtered_labels = filter_segments(chm_labels, chm_array_smooth, min_crown_area, min_circularity)

    array2raster(
        np.array(filtered_labels),
        os.path.join(output_dir, basename + '_labels_filtered.tif'),
        chm_array_metadata
        )

    tree_tops = local_maxima_to_points(
        local_maxi_coords,
        chm_array_smooth,
        metadata=chm_array_metadata
        )
    tree_tops.to_file(os.path.join(output_dir, basename + '_tree_tops.shp'))


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
    dataset = gdal.Open(geotif_file)
    if dataset.RasterCount != 1:
        raise ValueError('Function only supports single band data')

    metadata = {}
    metadata['RasterYSize'] = dataset.RasterYSize # Not the size of pixel, but size of the raster in pixels
    metadata['RasterXSize'] = dataset.RasterXSize
    metadata['driver'] = dataset.GetDriver().LongName
    metadata['projection'] = dataset.GetProjection()
    metadata['epsg'] = int(osr.SpatialReference(wkt=dataset.GetProjection()).GetAttrValue('AUTHORITY',1))
    metadata['geotransform'] = dataset.GetGeoTransform()

    metadata['pixelWidth'] = metadata['geotransform'][1]
    metadata['pixelHeight'] = metadata['geotransform'][5]

    metadata["extent_dict"] = {}
    metadata["extent_dict"]["xMin"] = metadata['geotransform'][0]
    metadata["extent_dict"]["xMax"] = metadata['geotransform'][0] + dataset.RasterXSize * metadata['geotransform'][1]
    metadata["extent_dict"]["yMin"] = metadata['geotransform'][3] + dataset.RasterYSize * metadata['geotransform'][5]
    metadata["extent_dict"]["yMax"] = metadata['geotransform'][3]

    metadata["extent"] = (
        metadata["extent_dict"]["xMin"],
        metadata["extent_dict"]["xMax"],
        metadata["extent_dict"]["yMin"],
        metadata["extent_dict"]["yMax"],
    )

    raster = dataset.GetRasterBand(1)
    metadata['noDataValue'] = raster.GetNoDataValue() if raster.GetNoDataValue() else 0
    metadata['scaleFactor'] = raster.GetScale()

    metadata['bandstats'] = {}
    stats = raster.GetStatistics(True, True)
    metadata["bandstats"]["min"] = round(stats[0], 2)
    metadata["bandstats"]["max"] = round(stats[1], 2)
    metadata["bandstats"]["mean"] = round(stats[2], 2)
    metadata["bandstats"]["stdev"] = round(stats[3], 2)

    array = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
    array[array == int(metadata["noDataValue"])] = np.nan
    array = array / metadata["scaleFactor"]
    return array, metadata


def array2raster(array, file_path, metadata):
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
    cols = array.shape[1]
    rows = array.shape[0]

    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(file_path, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((metadata['extent_dict']['xMin'], metadata['pixelWidth'], 0, 
                               metadata['extent_dict']['yMax'], 0, metadata['pixelHeight']))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
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
        Minimum crown area threshold in pixels (default: 10).
    min_circularity : float, optional
        Minimum circularity to keep a segment (default: 0). 

    Returns
    -------
    numpy.ndarray
        Filtered and relabeled segments array where segments not meeting
        the criteria have been removed.
    """
    props = regionprops(labels, intensity_image=chm_array)
    filtered_labels = labels.copy()
    
    for prop in props:

        crown_area = prop.area
        perimeter = prop.perimeter
        circularity = (4 * np.pi * crown_area) / (perimeter ** 2) if perimeter > 0 else 0

        # Apply filtering criteria
        if (crown_area < min_crown_area or circularity < min_circularity):
            filtered_labels[filtered_labels == prop.label] = 0

    filtered_labels = ndi.label(filtered_labels > 0)[0]
    return filtered_labels


def local_maxima_to_points(local_maxi_coords, chm_array, metadata):
    """
    Convert local maxima coordinates to a GeoDataFrame with tree heights
    
    Parameters:
    -----------
    local_maxi_coords : numpy.ndarray
        Array of (row, col) coordinates of local maxima.
    chm_array : numpy.ndarray
        The CHM array containing height values.
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

    heights = chm_array[local_maxi_coords[:, 0], local_maxi_coords[:, 1]]
    
    x_coords = geotransform[0] + local_maxi_coords[:, 1] * geotransform[1]
    y_coords = geotransform[3] + local_maxi_coords[:, 0] * geotransform[5]
    
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


# def raster2polygons(raster_array, metadata, output_file):
    """
    Converts a labeled raster array into polygon features.

    Parameters
    ----------
    raster_array : numpy.ndarray
        2D array where each unique value (except 0) represents a distinct polygon label.
    metadata : dict
        Metadata dictionary with spatial reference and geotransform info.
    output_file : str
        Path to save the output polygon shapefile.
    
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of polygons.
    """
    geotransform = metadata['geotransform']
    projection = metadata['projection']

    polygons = []
    labels = []
    
    for region in regionprops(raster_array):
        if region.label == 0:
            continue
        pass
    
    # Create GeoDataFrame and save to file
    gdf = gpd.GeoDataFrame({'label': labels, 'geometry': polygons}, crs=projection)
    gdf = gdf[gdf.is_valid]  # Remove invalid geometries
    gdf.to_file(output_file)

    return gdf


if __name__ == "__main__":
    run_segmentation()