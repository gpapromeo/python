import numpy as np
import pandas as pd
import gym
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord

INT_CLOCK = 1672531200  # 01.01.2023


class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, name='UNKNWON SAT'):
        # Initiate orbit object
        self.orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
        self.name = name    # satellite name
        self.int_clock = INT_CLOCK  # internal clock
        self.memory = list()

    def propagate(self, time):
        self.orb = self.orb.propagate(time * u.s, method='ValladoPropagator')   # propagate position
        return None

    def skycoord(self):
        satx, saty, satz = self.orb.r    # coordinates (cartesian reference)
        # create coordinate object
        sat_sc = SkyCoord(x=satx, y=saty, z=satz,
                          representation_type='cartesian')
        sat_sc.representation_type = 'spherical'
        sat_sc.ra.wrap_angle = 180 * u.degree  # angles +/-180deg
        sat_sc.dec.wrap_angle = 90 * u.degree  # angles +/-90deg
        return sat_sc

    def rcv_data(self, node_name, msg_ID, msg_payload):
        self.memory.append((node_name, msg_ID, msg_payload))

    def dwnl_data(self):
        memory = self.memory
        self.memory = list()
        return memory


class ground_station:
    def __init__(self, lat, lon, alt=1, name='UNKNOWN GS'):
        """
        altitude in meters
        """
        self.lat = lat  # latitude earth fixed
        self.lon = lon  # longitude earth fixed
        self.lon_IRF = lon  # longitude in inertial reference frame
        self.alt = alt  # altitude in meters. Must be greater than 0

        # Create the coordinate objects
        self.gs = coord.EarthLocation.from_geodetic(lon=lon*u.degree,
                                                    lat=lat*u.degree)
        self.gs_skycoord = SkyCoord(ra=self.gs.lon,   # ra = longitude
                                    dec=self.gs.lat,  # dec = latitude
                                    distance=6371*u.km + alt*u.m)
        self.gs_skycoord_cartesian = copy.deepcopy(self.gs_skycoord)
        self.gs_skycoord_cartesian.representation_type = 'cartesian'
        self.name = name    # Ground Station name
        self.int_clock = INT_CLOCK  # internal clock

    def skycoord(self):
        return self.gs_skycoord

    def skycoord_cartesian(self):
        return self.gs_skycoord_cartesian

    def propagate(self, time):
        self.int_clock += time  # update internal clock
        # update longitude due to earth rotation.
        self.lon_IRF = (self.lon_IRF + time/(24*60*60) * 360) % 360
        # update coordinate object
        self.gs = coord.EarthLocation.from_geodetic(lon=self.lon_IRF*u.degree,
                                                    lat=self.lat*u.degree)
        self.gs_skycoord = SkyCoord(ra=self.gs.lon,   # ra = longitude
                                    dec=self.gs.lat,  # dec = latitude
                                    distance=6371*u.km + self.alt*u.m)
        self.gs_skycoord.ra.wrap_angle = 180 * u.degree
        self.gs_skycoord.dec.wrap_angle = 90 * u.degree

        self.gs_skycoord_cartesian = self.gs_skycoord
        self.gs_skycoord_cartesian.representation_type = 'cartesian'
        return None

    def transmission(self):
        return None

