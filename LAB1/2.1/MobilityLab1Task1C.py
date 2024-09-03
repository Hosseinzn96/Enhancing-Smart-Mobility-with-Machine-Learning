#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymongo as pm
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import datetime
import time
import warnings
from shapely.geometry import Point, Polygon, shape
import seaborn as sns



client = pm.MongoClient('bigdatadb.polito.it',
 ssl=True,
 authSource = 'carsharing',
 username = 'ictts',
 password ='Ict4SM22!',
 tlsAllowInvalidCertificates=True)
db = client['carsharing']
"""Accesing at collections"""

PB_C2 = db['PermanentBookings']         
AB_C2 = db['ActiveBookings']            
PP_C2 = db['PermanentParkings']         
AP_C2 = db['ActiveParkings']            
PB_EJ = db['enjoy_PermanentBookings']   
AB_EJ = db['enjoy_ActiveBookings']      
PP_EJ = db['enjoy_PermanentParkings']   
AP_EJ = db['enjoyActiveParkings']       


# In[ ]:





# In[21]:


import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_pipeline(init_time, final_time, city):
    pipeline = [
        {
            '$match': {
                '$and': [
                    {'init_time': {'$gte': init_time + (tzMilano * 60 * 60)}},
                    {'init_time': {'$lte': final_time + (tzMilano * 60 * 60)}},
                    {'city': city}
                ]
            }
        },
        {
            '$project': {
                '_id': 0,
                'city': 1,
                'duration': {"$divide": [{'$subtract': ["$final_time", "$init_time"]}, 60]}
            }
        },
        {
            '$group': {
                '_id': "$duration",
                'count': {"$sum": 1}
            }
        },
        {
            '$sort': {'_id': 1}
        }
    ]
    return pipeline

def process_query_result(query_result):
    df = pd.DataFrame(list(query_result))
    c_df = np.cumsum(df['count'])
    data_sum = np.sum(df['count'])
    c_df_prob = c_df / data_sum
    mean_value = np.sum(df['_id']) / len(df['_id'])
    return df, c_df, data_sum, c_df_prob, mean_value

def plot_cdf(ax, duration_data, cdf_data, title, x_label, y_label, legend_labels):
    for i in range(len(duration_data)):
        ax.plot(duration_data[i]['_id'], cdf_data[i], label=legend_labels[i])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(b=True, which='minor', color='xkcd:grey', linestyle=':')
    ax.set_xscale("log")

# Time shift for Van
tzMilano = -1

# Time intervals for each week
week_intervals = [
    ('2017-11-01 00:00:00', '2017-11-08 23:59:59'),
    ('2017-12-11 00:00:00', '2017-12-18 23:59:59'),
    ('2018-01-01 00:00:00', '2018-01-08 23:59:59')
]

cities = ['Milano', 'Milano', 'Milano']  # Assuming the same city for each week

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Process each week
for i, (start_time, end_time) in enumerate(week_intervals, 1):
    init_time = time.mktime(datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').timetuple())
    final_time = time.mktime(datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').timetuple())

    pipeline = create_pipeline(init_time, final_time, cities[i - 1])

    # Process queries and DataFrames for each week
    query_B =PB_EJ.aggregate(pipeline)
    query_P =PP_EJ.aggregate(pipeline)

    Df_B, C_df_B, B_sum, C_df_B_prob, mean_B = process_query_result(query_B)
    Df_P, C_df_P, P_sum, C_df_P_prob, mean_P = process_query_result(query_P)

    # Plot CDF for Booking and Parking duration
    plot_cdf(axs[0], [Df_B], [C_df_B_prob], f"Booking duration for {cities[i - 1]} -{i} Weeks ",
             "", "CDF", [f"W{i}"])

    plot_cdf(axs[1], [Df_P], [C_df_P_prob], f"Parking duration for {cities[i - 1]} -{i} Weeks",
             "Duration [m]", "CDF", [f"W{i}"])

plt.tight_layout()
plt.show()


# In[ ]:




