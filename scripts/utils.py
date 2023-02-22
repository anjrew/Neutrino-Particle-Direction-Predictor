import os
import random
from typing import Tuple
import numpy as np
import math

def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    


def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
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
    print('r', r, 'x2y2', x2y2)
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


