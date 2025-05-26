#!/usr/bin/env python3

import math
import numpy
import pymap3d
from tf_transformations import euler_from_quaternion

def quaternion_to_euler(quat):
    '''
    Convert ROS Quaternion message to Euler angle representation (roll, pitch, yaw).

    :param quat: quaternion

    :return euler: roll=euler[0], pitch=euler[1], yaw=euler[2]
    '''
    q = [quat.x, quat.y, quat.z, quat.w]
    euler = euler_from_quaternion(q)
    return euler


def gps_to_enu(lat, lon, alt=0):
    '''
    Convert GPS coordinates (lat, lon, alt) to ENU coordinates (x, y, z).

    :param lat: Latitude in degrees
    :param lon: Longitude in degrees
    :param alt: Altitude in meters

    :return x, y, z: ENU coordinates in meters
    '''
    # Local coordinate origin (Sydney International Regatta Centre, Australia)
    lat0 = -33.724223 # degree North
    lon0 = 150.679736 # degree East
    alt0 = 0 # meters
    enu = pymap3d.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
    x = enu[0]
    y = enu[1]
    z = enu[2]
    return x, y, z
