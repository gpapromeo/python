from astropy import units as u
from poliastro.bodies import Earth, Sun
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter3D
import plotly.io as pio
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import numpy as np


class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, name='UNKNWON SAT'):
        # Initiate orbit object
        self.orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
        self.name = name    # satellite name
        self.int_clock = 0  # internal clock

    def propagate(self, time):
        self.orb = self.orb.propagate(time * u.s)   # propagate position
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


class ground_station:
    def __init__(self, lat, lon, alt=0, name='UNKNOWN GS'):
        """
        altitude in meters
        """
        self.lat = lat  # latitude
        self.lon = lon  # longitude
        self.alt = alt  # altitude in meters

        # Create the coordinate objects
        self.gs = coord.EarthLocation.from_geodetic(lon=lon*u.degree,
                                                    lat=lat*u.degree)
        self.gs_skycoord = SkyCoord(ra=self.gs.lon,   # ra = longitude
                                    dec=self.gs.lat,  # dec = latitude
                                    distance=6371*u.km + alt*u.m)
        self.name = name    # Ground Station name
        self.int_clock = 0  # internal clock

    def skycoord(self):
        return self.gs_skycoord

    def propagate(self, time):
        # update longitude due to earth rotation.
        self.lon = (self.lon + time/(24*60*60) * 360) % 360
        # update coordinate object
        self.gs = coord.EarthLocation.from_geodetic(lon=self.lon*u.degree,
                                                    lat=self.lat*u.degree)
        self.gs_skycoord = SkyCoord(ra=self.gs.lon,   # ra = longitude
                                    dec=self.gs.lat,  # dec = latitude
                                    distance=6371*u.km + self.alt*u.m)
        self.gs_skycoord.ra.wrap_angle = 180 * u.degree
        self.gs_skycoord.dec.wrap_angle = 90 * u.degree
        return None

    def transmission(self):
        return None


class sched_transm_gs(ground_station):
    def __init__(self, lat, lon, alt=0, name='UNKNOWN GS'):
        super().__init__(lat, lon, alt, name)
        self.sched_time = 60 * 10   # schedul transm interval in seconds
        self.repeat = 3
        self.repeat_count = 0

    def dgp():
        return np.random.randint(0,25)

    def transmission(self):
        if self.repeat_count > 0:
            self.repeat_count -= 1
            return data
        if self.int_clock > self.sched_time:


N_constellation = 4
sats = list()

a = 6928 * u.km
ecc = 0 * u.one
inc = 97.59 * u.deg
raan = 270 * u.deg
argp = 0 * u.deg
nu = -180 * u.deg

sats = list()
for i in range(N_constellation):
    sats.append(satellite(a, ecc, inc, raan, argp,
                          nu + 360*i/N_constellation * u.deg,
                          name='PICO'+str(i)))

gs = ground_station(45.5, 10.2, 0, 'Brescia 01')

t_stp = 60  # seconds
sim_steps = 200
for stp in range(sim_steps):
    gs.propagate(t_stp)
    for sat in sats:
        sat.propagate(t_stp)
        link_angl = gs.skycoord().separation(sat.skycoord())
        if link_angl <= 70 * u.degree:
            print(f'connection with satellite {sat.name}')
