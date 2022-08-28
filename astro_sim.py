from msilib.schema import Error
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
# from poliastro.plotting import OrbitPlotter3D
from poliastro.core.events import line_of_sight
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

INT_CLOCK = 1672531200  # 01.01.2023


class satellite:
    def __init__(self, a, ecc, inc, raan, argp, nu, name='UNKNWON SAT'):
        # Initiate orbit object
        self.orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
        self.name = name    # satellite name
        self.int_clock = INT_CLOCK  # internal clock
        self.memory = list()

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


class sched_transm_gs(ground_station):
    def __init__(self, lat, lon, alt=0, name='UNKNOWN GS'):
        super().__init__(lat, lon, alt, name)
        self.sched_time = 60 * 10   # schedul transm interval in seconds
        self.last_trnsmt = self.int_clock
        self.repeat = 0     # Number of message repeats
        self.repeat_count = 0   # Counter for repeated messages
        self.msg_id = -1     # message ID

    def dgp(self):      # Data Generation Process
        return np.random.randint(0, 25)

    def transmission(self):
        if self.int_clock - self.last_trnsmt > self.sched_time:
            # Run this every sched_time seconds
            self.msg_id += 1
            self.last_trnsmt = self.int_clock
            self.data = self.dgp()
            self.repeat_count = self.repeat
            return self.data, self.msg_id
        elif self.repeat_count > 0:
            # Run this when the repeat counter is positive
            # (i.e. we repeat the last transmission)
            self.repeat_count -= 1
            return self.data, self.msg_id
        else:
            return None, None


class master_gs(ground_station):
    def __init__(self, lat, lon, alt=0, name='UNKNOWN MASTER GS'):
        super().__init__(lat, lon, alt, name)
        self.data = dict()      # downloaded data
        # For each satellite a list with downloaded data
        # is stored.

    def rcv_data(self, sat_name, data):
        if sat_name not in self.data.keys():
            self.data.update({sat_name: list()})
        self.data[sat_name].extend(data)

    def n_sat_received(self):
        return len(self.data)

    def clear_memory(self):
        self.data = dict()


class policy_nn(nn.Module):
    def __init__(self, nsat, ngs):
        super(policy_nn, self).__init__()
        self.main = nn.Sequential(
            nn.Linear((nsat, ngs), (nsat, ngs, 64)),
            nn.ReLU(),
            nn.Linear((nsat, ngs, 64), (nsat, ngs)),
        )

    def forward(self, input):
        return self.main(input)


class constellation_env:
    def __init__(self, mgs_lat, mgs_lon, mgs_alt, mgs_name):
        self.sats = list()
        self.gss = list()
        self.mgs = master_gs(mgs_lat, mgs_lon, mgs_alt, mgs_name)

    def reset(self):
        self.sat_names = [sat.name for sat in self.sats]
        self.data_lst = list()
        # TODO reset satellite clock and msg counters
        # TODO reset ground stations clock and msg counters
        self.observation = self.sym_step()
        return self.observation

    def add_gs_lst(self, gs_coord_lst):
        self.gs_coord_lst = gs_coord_lst

        for i in range(len(gs_coord_lst)):
            self.gss.append(sched_transm_gs(gs_coord_lst[i][0],
                                            gs_coord_lst[i][1],
                                            100, 'GS' + str(i)))

    def add_uniform_const(self, N_constellation, a, ecc, inc,
                          raan, argp, nu):
        for i in range(N_constellation):
            self.sats.append(satellite(a, ecc, inc, raan, argp,
                                       nu + 360*i/N_constellation * u.deg,
                                       name='PICO'+str(i)))

    def sym_step(self):
        t_stp = 60  # seconds
        data_row = {'msg_id': -1, 'data': np.nan, 'time': np.nan,
                    'latitude': np.nan, 'longitude': np.nan}
        [data_row.update({sat.name: 0}) for sat in self.sats]
        keys = list(data_row.keys())

        while True:
            for sat in self.sats:
                sat.propagate(t_stp)    # propagate all satellites
            for gs in self.gss:
                gs.propagate(t_stp)     # propagate all ground stations nodes
            self.mgs.propagate(t_stp)        # propagate master ground station

            mgs_sc = self.mgs.skycoord()
            for sat in self.sats:
                los_evnt = line_of_sight([mgs_sc.x, mgs_sc.y, mgs_sc.z] * u.km,
                                         sat.orb.r, 6371*u.km)
                if los_evnt >= 0:
                    self.mgs.rcv_data(sat.name, sat.dwnl_data())

            for gs in self.gss:
                gs_sc = gs.skycoord()
                payload, id = gs.transmission()
                data_tmp_row = copy.deepcopy(data_row)
                data_tmp_row.update({'latitude': gs.lat, 'longitude': gs.lon,
                                    'time': gs.int_clock})
                for sat in self.sats:
                    los_evnt = line_of_sight([gs_sc.x, gs_sc.y, gs_sc.z] * u.km,
                                             sat.orb.r, 6371*u.km)
                    if los_evnt >= 0 and payload is not None:
                        # The satellite received data!
                        sat.rcv_data(gs.name, id, payload)
                        distance = gs_sc.separation_3d(sat.skycoord())
                        data_tmp_row.update({'msg_id': id, sat.name: distance})
                if data_tmp_row['msg_id'] != -1:
                    dicti = copy.deepcopy(data_tmp_row)
                    for key in keys:
                        if key not in self.sat_names:
                            dicti.pop(key)
                        elif dicti[key] == 0:
                            dicti.pop(key)
                    dicti_values = {key: i for (key, _), i in
                                    zip(sorted(dicti.items(),
                                               key=lambda item: item[1]),
                                        range(1, len(dicti)+1))}
                    data_tmp_row.update(dicti_values)
                    self.data_lst.append(data_tmp_row)
            if self.mgs.n_sat_received() == len(self.sats):
                data = self.mgs.data
                m = data.values()
                obs = np.zeros((len(m), len(self.gs_coord_lst)))
                for m_sat, sat_id in zip(m, range(len(m))):
                    for msg_m in m_sat:
                        obs[sat_id, int(msg_m[0][2])] += 1
                self.mgs.clear_memory()
                return obs

    def policy_reward(self, action):
        if len(action) != len(self.sat_names):
            raise Error
        elif len(action[0]) != len(self.gs_coord_lst):
            raise Error
        else:
            gs_sum = np.sum(action, axis=0)
            individual_gs = np.array(gs_sum > 0, dtype=int)
            return np.sum(individual_gs)

    def step(self, action):
        reward = self.policy_reward(action)
        self.observation = self.sym_step()   # New data
        return self.observation, reward

    def save_data_lst(self):
        df = pd.DataFrame(self.data_lst)
        # sdf = df.groupby('msg_id').sum()
        pd.to_pickle(df, 'dgp_contacts.pkl')


const = constellation_env(45.45, 10.25, 100, 'Master Brescia GS')

# Adding satellites to the constellation
N_constellation = 4
a = 6928 * u.km
ecc = 0 * u.one
inc = 97.59 * u.deg
raan = 270 * u.deg
argp = 0 * u.deg
nu = -180 * u.deg
const.add_uniform_const(N_constellation, a, ecc, inc, raan, argp, nu)

# Adding ground stations nodes.
gs_coord_lst = [[55.5, 10.2], [50, 8], [70, -110], [75, -112]]
const.add_gs_lst(gs_coord_lst)

# Reset environment
const.reset()

# Training policy


# Start demonstration 
for _ in range(10):
    action = [np.random.randint(0, 2, 4),
              np.random.randint(0, 2, 4),
              np.random.randint(0, 2, 4),
              np.random.randint(0, 2, 4)]
    observation, reward, = const.step(action)
