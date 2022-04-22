"""
Author:         Romeo Casesa
First issue:    14-04-2022
Last modified:  14-04-2022

This file simulates the behaviour of the satellite and the data reception on ground.
"""
import numpy as np
import pandas as pd
import datetime
from collections import deque

cust_df = pd.read_pickle('customer_node_data.pkl')
cust_queue = dict()
cu_ids = cust_df.loc[:, 'CU'].unique()
for cu_id in cu_ids:
    tmp_cu = cust_df.loc[cust_df.loc[:, 'CU'] == cu_id, ]
    tmp_cu = tmp_cu.drop(columns='CU')
    tmp_cu_lst = tmp_cu.to_dict('records')
    tmp_cu_queue = deque(tmp_cu_lst)
    cu_id_str = 'CU' + str(cu_id)
    cust_queue.update({cu_id_str: tmp_cu_queue})


contacts = pd.read_pickle('contact_events.pkl')
sats = contacts.loc[:, 'satellite'].unique()
sats_queue = dict()
[sats_queue.update({sat_id: deque()}) for sat_id in sats]

for i in range(len(contacts)):
    contact = contacts.loc[i, ]
    if contact['duration'] > datetime.timedelta(seconds=20):
        if contact['link'] == 'Downlink':
            pass
        elif contact['link'] == 'Uplink':
            pass
        else:
            pass
