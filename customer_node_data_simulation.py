"""
Author:         Romeo Casesa
First issue:    13-04-2022
Last modified:  14-04-2022

This file simulates data from IoT nodes.
It is assumed that IoT nodes transmit at fixed intervals
"""
import numpy as np
import pandas as pd
import datetime


def create_mess(i, transm_time):
    CU_id = bytes(str(i), 'ASCII')
    time = bytes(str(transm_time), 'ASCII')
    reply = b'FEES2-' + CU_id + time
    return reply


# time between one transmission and the next
CU_freq = [900, 3600, 11400, 180]       # [sec]
CU_td = [datetime.timedelta(seconds=td) for td in CU_freq]

N_CU = len(CU_freq)         # Number of client nodes

# Simulation start time. Must coincide with GMAT simulation.
sim_start = datetime.datetime(2023, 1, 1, 0, 0, 0, 0)
# Simulation end time.
sim_end = datetime.datetime(2023, 3, 1, 0, 0, 0, 0)
# Initialize dataframe
df_dict = []
for i in range(N_CU):
    # Number of transmission attempts within the simulation time.
    # Assumes transmissions are equally distributed in time.
    transm_attempts = (sim_end - sim_start) // CU_td[i]
    transm_time = sim_start
    for ta in range(transm_attempts):
        dt = {}
        transm_time = (transm_time + CU_td[i]
                       + datetime.timedelta(0, np.random.randint(-120, 120)))
        dt.update({'CU': i})
        dt.update({'timestamp': transm_time})
        dt.update({'message': create_mess(i, transm_time)})
        df_dict.append(dt)
df = pd.DataFrame(df_dict)
df.to_pickle('customer_node_data.pkl')
