import os
import random
from typing import Tuple
import numpy as np
import pandas as pd
import math
import logging

def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    


def angular_dist_score(az_true, zen_true, az_pred, zen_pred) -> float:
    '''
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse 
    cosine (arccos) thereof is then the angle between the two input vectors
    
    The lower the angle, the more similar the two vectors are meaning the score is better.
    
    Parameters:
    -----------
    
    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian
    
    Returns:
    --------
    
    dist : float
        mean over the angular distance(s) in radian
    '''
    
    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")
    
    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)
    
    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)
    
    # scalar product of the two Cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    # scalar product of two unit vectors is always between -1 and 1, this is against numerical instability
    # that might otherwise occur from the finite precision of the sine and cosine functions
    scalar_prod =  np.clip(scalar_prod, -1, 1)
    
    # convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))


def cartesian_to_sphere(x: float, y: float, z: float) -> Tuple[float, float]:
    """Maps vector cartesian coordinates (x, y, z) from the origin to spherical angles azimuth and zenith.
    
    See: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        z (float): The z-coordinate of the point.

    Returns:
        tuple[float, float]: The azimuth and zenith angles in radians.
    """
    x2y2 = x**2 + y**2
    r = math.sqrt(x2y2 + z**2)
    azimuth = math.acos(x / math.sqrt(x2y2)) * np.sign(y)
    zenith = math.acos(z / r)
    return azimuth, zenith


def sphere_to_cartesian(azimuth: float, zenith: float) -> Tuple[float, float, float]:
    """Map spherical coordinates to cartesian coordinates.
    see: https://stackoverflow.com/a/10868220/4521646
    
    Args:
        azimuth (float): The azimuth angle in radians.
        zenith (float): The zenith angle in radians.

    Returns:
        tuple: The x, y, z vector cartesian coordinates of the point from the origin.
    """
    x = math.sin(zenith) * math.cos(azimuth)
    y = math.sin(zenith) * math.sin(azimuth)
    z = math.cos(zenith)
    return x, y, z


def adjust_sphere(azimuth:float, zenith:float) -> Tuple[float, float]:
    """Adjust azimuth and zenith to be within [-pi, pi]

    Args:
        azimuth (float): The azimuth to adjust
        zenith (float): The zenith to adjust

    Returns:
        float: The adjusted azimuth and zenith
    """
    if zenith < 0:
        zenith += math.pi
        azimuth += math.pi
    if azimuth < 0:
        azimuth += math.pi * 2
    azimuth = azimuth % (2 * math.pi)
    return azimuth, zenith



def compose_event_df(
    batch_df: pd.DataFrame,
    event_id: str,
    sensor_geometry: pd.DataFrame,
) -> pd.DataFrame:
    """Composes the event dataframe from the batch dataframe and the sensor geometry dataframe.

    Args:
        batch_df (pd.DataFrame): _description_
        event_id (str): The event id to extract from the data frame
        sensor_geometry (pd.DataFrame): The dataframe containing the sensor geometry.

    Returns:
        pd.DataFrame: a dataframe containing the data for one event.
    """
    
    # filter the batch dataframe for the event_id
    event_df = batch_df[batch_df['event_id'] == event_id]
    # merge the event dataframe with the sensor geometry dataframe
    event_df = pd.merge(
        left = event_df,
        right = sensor_geometry,
        how='inner',
        on='sensor_id'
    )
    # sort the dataframe by sensor_id
    event_df.sort_values(by=['time'], inplace=True)
    return event_df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        logging.info(f'Optimizing col {col}')
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    logging.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logging.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df




def convert_bytes_to_gmbkb(bytes) -> str:
    """Converts bytes to a human readable format."""
    if bytes >= 1099511627776:  # 1 TB = 1024 GB * 1024 MB * 1024 KB bytes
        return f"{bytes / 1099511627776:.2f} TB"
    elif bytes >= 1073741824:  # 1 GB = 1024 MB * 1024 KB bytes
        return f"{bytes / 1073741824:.2f} GB"
    elif bytes >= 1048576:  # 1 MB = 1024 KB bytes
        return f"{bytes / 1048576:.2f} MB"
    elif bytes >= 1024:  # 1 KB = 1024 bytes
        return f"{bytes / 1024:.2f} KB"
    else:
        return f"{bytes} bytes"

