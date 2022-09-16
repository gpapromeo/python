import numpy as np
import pandas as pd
import copy
import random
import gym

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.core.events import line_of_sight

from ray.rllib.env.multi_agent_env import MultiAgentEnv


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

    @property
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
    
    def reset(self):
        return None       


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

    @property
    def skycoord(self):
        # update coordinate object
        self.gs = coord.EarthLocation.from_geodetic(lon=self.lon_IRF*u.degree,
                                                    lat=self.lat*u.degree)
        self.gs_skycoord = SkyCoord(ra=self.gs.lon,   # ra = longitude
                                    dec=self.gs.lat,  # dec = latitude
                                    distance=6371*u.km + self.alt*u.m)
        self.gs_skycoord.ra.wrap_angle = 180 * u.degree
        self.gs_skycoord.dec.wrap_angle = 90 * u.degree
        return self.gs_skycoord

    @property
    def skycoord_cartesian(self):
        self.gs = coord.EarthLocation.from_geodetic(lon=self.lon_IRF*u.degree,
                                                    lat=self.lat*u.degree)
        self.gs_skycoord_cartesian = SkyCoord(ra=self.gs.lon,   # ra = longitude
                                              ec=self.gs.lat,  # dec = latitude
                                              distance=6371*u.km + self.alt*u.m)
        self.gs_skycoord_cartesian.ra.wrap_angle = 180 * u.degree
        self.gs_skycoord_cartesian.dec.wrap_angle = 90 * u.degree
        self.gs_skycoord_cartesian.representation_type = 'cartesian'
        return self.gs_skycoord_cartesian

    def propagate(self, time):
        self.int_clock += time  # update internal clock
        # update longitude due to earth rotation.
        self.lon_IRF = (self.lon_IRF + time/(24*60*60) * 360) % 360
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
    def __init__(self, N_nodes, lat, lon, alt=0,
                 name='UNKNOWN MASTER GS'):
        super().__init__(lat, lon, alt, name)
        self.data = dict()      # downloaded data
        # For each satellite a list with downloaded data
        # is stored.
        self.N_nodes = N_nodes

    def rcv_data(self, sat_name, data):
        if sat_name not in self.data.keys():
            self.data.update({sat_name: list()})
        self.data[sat_name].extend(data)

    def get_obs(self):
        obs = dict()
        for sat_name in self.data.keys():
            obs.update({sat_name: np.zeros(self.N_nodes)})
        for sat_name in self.data.keys():
            sat_mem = self.data[sat_name]
            for node_name, _, _ in sat_mem:
                node_id = int(node_name.split('_')[-1])
                sat_obs = obs['sat_name']
                sat_obs[node_id] = 1
                obs.update({sat_name: sat_obs})
        self.data = dict()
        return obs


class BasicMultiAgent(MultiAgentEnv):
    """Env of N independent agents, each of which exits after 25 steps.
    credit: https://github.com/ray-project/ray/blob/master/rllib/examples/env/multi_agent.py
    https://www.gymlibrary.dev/content/environment_creation/
    """

    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, N_constellation, N_nodes):
        super().__init__()
        self.N_nodes = N_nodes
        self.N_constellation = N_constellation

        a = 6928 * u.km * np.ones(N_constellation)
        ecc = 0 * u.one * np.ones(N_constellation)
        inc = 97.59 * u.deg * np.ones(N_constellation)
        raan = 270 * u.deg * np.ones(N_constellation)
        argp = 0 * u.deg * np.ones(N_constellation)
        nu = -180 * u.deg + np.linspace(0, 360, N_constellation) * u.deg
        self.agents = [satellite(a[i], ecc[i], inc[i], raan[i],
                                 argp[i], nu[i], name='PICO_'+str(i))
                       for i in range(N_constellation)]
        self._agent_ids = set(range(N_constellation))
        self.sat_names = [sat.name for sat in self.agents]
        self.mgs = master_gs(N_nodes, 45.45, 10.25, 100, 'Master Brescia GS')
        self.gss = [sched_transm_gs(np.random.randint(-89, 89),
                                    np.random.randint(-179, 180),
                                    np.random.randint(1, 1000),
                                    'node_' + str(i)) for i in range(N_nodes)]
        self.dones = set()
        self.observation_space = dict()
        [self.observation_space.update({sat_name: gym.spaces.MultiBinary(N_nodes)})
            for sat_name in self.sat_names]
        self.action_space = dict()
        [self.action_space.update({sat_name: gym.spaces.MultiBinary(N_nodes)})
            for sat_name in self.sat_names]
        self.resetted = False

    def reset(self, seed):
        super().reset(seed=seed)
        self.resetted = True
        self.gss = [sched_transm_gs(np.random.randint(-89, 89),
                                    np.random.randint(-179, 180),
                                    np.random.randint(1, 1000),
                                    'node_' + str(i)) for i in range(self.N_nodes)]
        self._simulate(t_stp=60)
        self.dones = set()
        # TODO Resetting agents is not doing anything. It should return a new observation.
        # return {i: a.reset() for i, a in enumerate(self.agents)}
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _simulate(self, t_stp):
        dones = dict.fromkeys(self.sat_names, 0)
        while sum(dones.values()) < self.N_constellation:
            for sat in self.agents:
                sat.propagate(t_stp)    # propagate all satellites
            for gs in self.gss:
                gs.propagate(t_stp)     # propagate all ground stations nodes
            self.mgs.propagate(t_stp)        # propagate master ground station
            mgs_sc = self.mgs.skycoord
            for sat in self.agents:
                los_evnt = line_of_sight([mgs_sc.x, mgs_sc.y, mgs_sc.z] * u.km,
                                        sat.orb.r, 6371*u.km)
            if los_evnt >= 0:
                self.mgs.rcv_data(sat.name, sat.dwnl_data())
                dones[sat.name] = 1
            for gs in self.gss:
                gs_sc = gs.skycoord
                payload, id = gs.transmission()
                for sat in self.sats:
                    los_evnt = line_of_sight([gs_sc.x, gs_sc.y, gs_sc.z] * u.km,
                                            sat.orb.r, 6371*u.km)
                    if los_evnt >= 0 and payload is not None:
                        # The satellite received data!
                        sat.rcv_data(gs.name, id, payload)


    def _get_info(self):
        return None

    def _get_obs(self):
        # Returns the current state observation
        return self.mgs.get_obs()

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info

    def render(self, mode="rgb_array"):
        # Just generate a random image here foNr demonstration purposes.
        # Also see `gym/envs/classic_control/cartpole.py` for
        # an example on how to use a Viewer object.
        return np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)

maenv = BasicMultiAgent(4, 6)