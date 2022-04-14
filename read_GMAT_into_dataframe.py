"""
Author:         Romeo Casesa
First issue:    14-04-2022
Last modified:  14-04-2022

This file reads the simulation output from GMAT and transmorms this
into a pandas dataframe.
"""

import os
import pandas as pd
import re
import datetime

root, dirs, files = next(os.walk("../results"))
df_dict = list()
for file in files:
    target = None
    observer = None
    with open(os.path.join(root, file)) as ff:
        if 'Uplink' in file:
            link = 'Uplink'
        elif 'Downlink' in file:
            link = 'Downlink'
        else:
            link = 'Unknown'
        for fl in ff:
            if re.findall('Target: (\w+)', fl):                 # noqa: W605
                target = re.findall('Target: (\w+)', fl)[0]     # noqa: W605
            elif re.findall('Observer: (\w+)', fl):             # noqa: W605
                observer = re.findall('Observer: (\w+)', fl)[0]    # noqa: W605
            elif re.findall("(\d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2}.\d{3})", fl):     # noqa: W605
                start_str, end_str = re.findall("(\d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2}.\d{3})", fl)   # noqa: W605
                start_dt = datetime.datetime.strptime(start_str,
                                                    '%d %b %Y %H:%M:%S.%f')
                end_dt = datetime.datetime.strptime(end_str,
                                                    '%d %b %Y %H:%M:%S.%f')
                duration = end_dt - start_dt
                dct = {'start': start_dt, 'end': end_dt, 'duration': duration,
                      'satellite': target, 'ground station': observer, 'link': link}
                df_dict.append(dct)
            else:
                pass

    df = pd.DataFrame(df_dict)
    df.sort_values(by=['start'], inplace=True)
    df = df.reset_index(drop=True)
    pd.to_pickle(df, 'contact_events.pkl')
