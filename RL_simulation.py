"""
Author:         Romeo Casesa
First issue:    14-04-2022
Last modified:  03-05-2022

This file simulates the behaviour of the satellite and the
data reception on ground.
"""
import numpy as np
import pandas as pd
import datetime
from collections import deque
from copy import deepcopy


def cust_df_to_queue(cust_df):
    """
    Create a dictionary with:
    keys: the different customers nodes (i.e. CU0, CU1 etc..)
    params: a two-sided queue with messages and timestamps
    """
    cust_queue = dict()     # Initialize dictionary
    cu_ids = cust_df.loc[:, 'CU'].unique()  # Unique customer ids

    for cu_id in cu_ids:
        # Filter one specific CU and drop column with CU id
        tmp_cu = cust_df.loc[cust_df.loc[:, 'CU'] == cu_id, ]
        tmp_cu = tmp_cu.drop(columns='CU')

        # Make list of dictionaries (one entry for each row of the dataframe)
        tmp_cu_lst = tmp_cu.to_dict('records')
        tmp_cu_queue = deque(tmp_cu_lst)
        cu_id_str = 'CU' + str(cu_id)

        cust_queue.update({cu_id_str: tmp_cu_queue})
        """
        Should result is something like:
            {'CU0': deque([{'timestamp': Timestamp('2023-01-01 00:15:56'),
                'message': b'FEES2-02023-01-01 00:15:56'},
                {'timestamp': Timestamp('2023-01-01 00:32:25'),
                'message': b'FEES2-02023-01-01 00:32:25'},
                ....]),
             'CU1': deque([...]),
             ... }
        """
    return cust_queue


def send_policy(satellite_queue):
    """
    Specify the policy for downlinking data
    """
    return


cust_df = pd.read_pickle('customer_node_data.pkl')
cust_queue = cust_df_to_queue(cust_df)

contacts = pd.read_pickle('contact_events.pkl')
sats = contacts.loc[:, 'satellite'].unique()

# each satellite has its own cu_queue where to pop items from. This is
# needed as the same communication can be received by multiple satellites.
sats_queue = dict()
[sats_queue.update({sat_id: {'onboard_queue': deque(),
                             "cu_queue": deepcopy(cust_queue)}}) for sat_id in sats]

for i in range(len(contacts)):
    contact = contacts.loc[i, ]
    breakme = False
    if contact['duration'] > datetime.timedelta(seconds=5):
        if contact['link'] == 'Downlink':
            send_policy(sats_queue[contact['satellite']]['onboard_queue'])
            sats_queue[contact['satellite']]['onboard_queue'].clear()  # empty the queue
        elif contact['link'] == 'Uplink':
            msg_que = sats_queue[contact['satellite']]['cu_queue'][contact['ground station']]
            try:
                msg = msg_que.popleft()
            except IndexError:
                continue
            if contact['end'] <= msg['timestamp']:
                msg_que.appendleft(msg)
                continue
            else:
                while msg['timestamp'] < contact['end']:
                    if msg['timestamp'] < contact['start']:
                        try:
                            msg = msg_que.popleft()
                        except IndexError:
                            breakme = True
                            break
                    else:
                        sats_queue[contact['satellite']]['onboard_queue'].append(msg)
                        try:
                            msg = msg_que.popleft()
                        except IndexError:
                            breakme = True
                            break
                if breakme:
                    break
                msg_que.appendleft(msg)
        else:
            pass
