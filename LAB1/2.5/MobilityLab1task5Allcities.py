#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import datetime
import time
import warnings
import pymongo as pm #import MongoClient only
client = pm.MongoClient('bigdatadb.polito.it',
 ssl=True,
 authSource = 'carsharing',
 username = 'ictts',
 password ='Ict4SM22!',
 tlsAllowInvalidCertificates=True)
db = client['carsharing']


# In[7]:


def print_result(data):
    print('{\n"Day" : ' + str(data["_id"]) + ',\n"Total Rentals" : ' + str(data["count"]) +
        ',\n"Average" : ' + str(data["avg"]) + ',\n"Std Dev" : ' + str(data["stdDev"]) +
        ',\n"Median" : ' + str(data["median"]) + ',\n"P25" : ' + str(data["p25"]) +
        ',\n"P75" : ' + str(data["p75"]) + '\n}')

# Define start and end dates
start_date = datetime.datetime(2017, 12, 1, 0, 0, 0)
end_date = datetime.datetime(2017, 12, 31, 23, 59, 59)

# Convert dates to Unix time
start_unix_time = int(start_date.timestamp())
end_unix_time = int(end_date.timestamp())

# Generate date range
date_range = [{"day": day} for day in range(1, 32)]  # Assuming 1 to 31 for simplicity

# MongoDB aggregation pipeline
pipeline = [
    {
        "$match": {
            "init_time": {"$gte": start_unix_time, "$lte": end_unix_time},
            "city": "Vancouver"
        }
    },
    {
        "$project": {
            "_id": 0,
            "city": 1,
            "init_date": 1,
            "duration": {"$ceil": {"$divide": [{"$subtract": ["$final_time", "$init_time"]}, 60]}},
            "moved": {
                "$ne": [
                    {"$arrayElemAt": ["$origin_destination.coordinates", 0]},
                    {"$arrayElemAt": ["$origin_destination.coordinates", 1]}
                ]
            }
        }
    },
    {
        "$match": {
            "duration": {"$gte": 3, "$lte": 150}
        }
    },
    {
        "$project": {
            "city": 1,
            "duration": 1,
            "moved": 1,
            "day": {"$dayOfMonth": "$init_date"}
        }
    },
    {
        "$group": {
            "_id": "$day",
            "count": {"$sum": 1},
            "avg": {"$avg": "$duration"},
            "stdDev": {"$stdDevPop": "$duration"},
            "durations": {"$push": "$duration"}
        }
    },
    {
        "$sort": {"_id": 1}
    },
    {
        "$project": {
            "_id": 1,
            "count": 1,
            "avg": 1,
            "stdDev": 1,
            "median":{"$arrayElemAt":["$durations",
              {"$floor":{ "$multiply": [0.5,{"$size": "$durations"}]}}]},
            "p25": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.25, {"$size": "$durations"}]}}]},
            "p75": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.75, {"$size": "$durations"}]}}]}
        }
    }
]

# Execute the pipeline
result_cursor = db.PermanentBookings.aggregate(pipeline)
result_list = list(result_cursor)

# Print the results
flag = True
for i, date in enumerate(date_range):
    existing_data = next((data for data in result_list if data["_id"] == date["day"]), {
        "_id": date["day"],
        "count": 0,
        "avg": 0,
        "stdDev": 0,
        "median": 0,
        "p25": 0,
        "p75": 0
    })

    if flag:
        print('[')
        flag = False

    # Moved the print_result function call here
    print_result(existing_data)

    if i < len(date_range) - 1:
        print(',')

print(']')
import matplotlib.pyplot as plt

# Execute the pipeline
result_cursor = db.PermanentBookings.aggregate(pipeline)
result_list = list(result_cursor)

# Extract relevant data for plotting
days = [data["_id"] for data in result_list]
avg_values = [data["avg"] for data in result_list]
stdDev_values = [data["stdDev"] for data in result_list]
median_values = [data["median"] for data in result_list]
p25_values = [data["p25"] for data in result_list]
p75_values = [data["p75"] for data in result_list]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(days, avg_values, label='Average')
plt.plot(days, stdDev_values, label='Standard Deviation')
plt.plot(days, median_values, label='Median')
plt.plot(days, p25_values, label='P25')
plt.plot(days, p75_values, label='P75')

plt.xlabel('Day')
plt.ylabel('Value')
plt.title('Vancouver(Booking) stats on December ')
plt.legend()
plt.grid(True)
plt.show()


# In[8]:


def print_result(data):
    print('{\n"Day" : ' + str(data["_id"]) + ',\n"Total Rentals" : ' + str(data["count"]) +
        ',\n"Average" : ' + str(data["avg"]) + ',\n"Std Dev" : ' + str(data["stdDev"]) +
        ',\n"Median" : ' + str(data["median"]) + ',\n"P25" : ' + str(data["p25"]) +
        ',\n"P75" : ' + str(data["p75"]) + '\n}')

# Define start and end dates
start_date = datetime.datetime(2017, 12, 1, 0, 0, 0)
end_date = datetime.datetime(2017, 12, 31, 23, 59, 59)

# Convert dates to Unix time
start_unix_time = int(start_date.timestamp())
end_unix_time = int(end_date.timestamp())

# Generate date range
date_range = [{"day": day} for day in range(1, 32)]  # Assuming 1 to 31 for simplicity

# MongoDB aggregation pipeline
pipeline = [
    {
        "$match": {
            "init_time": {"$gte": start_unix_time, "$lte": end_unix_time},
            "city": "Vancouver"
        }
    },
    {
        "$project": {
            "_id": 0,
            "city": 1,
            "init_date": 1,
            "duration": {"$ceil": {"$divide": [{"$subtract": ["$final_time", "$init_time"]}, 60]}},
            "moved": {
                "$eq": [
                    {"$arrayElemAt": ["$origin_destination.coordinates", 0]},
                    {"$arrayElemAt": ["$origin_destination.coordinates", 1]}
                ]
            }
        }
    },
    {
        "$match": {
            "duration": {"$gte": 3, "$lte": 150}
        }
    },
    {
        "$project": {
            "city": 1,
            "duration": 1,
            "moved": 1,
            "day": {"$dayOfMonth": "$init_date"}
        }
    },
    {
        "$group": {
            "_id": "$day",
            "count": {"$sum": 1},
            "avg": {"$avg": "$duration"},
            "stdDev": {"$stdDevPop": "$duration"},
            "durations": {"$push": "$duration"}
        }
    },
    {
        "$sort": {"_id": 1}
    },
    {
        "$project": {
            "_id": 1,
            "count": 1,
            "avg": 1,
            "stdDev": 1,
            "median":{"$arrayElemAt":["$durations",
              {"$floor":{ "$multiply": [0.5,{"$size": "$durations"}]}}]},
            "p25": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.25, {"$size": "$durations"}]}}]},
            "p75": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.75, {"$size": "$durations"}]}}]}
        }
    }
]

# Execute the pipeline
result_cursor = db.PermanentParkings.aggregate(pipeline)
result_list = list(result_cursor)

# Print the results
flag = True
for i, date in enumerate(date_range):
    existing_data = next((data for data in result_list if data["_id"] == date["day"]), {
        "_id": date["day"],
        "count": 0,
        "avg": 0,
        "stdDev": 0,
        "median": 0,
        "p25": 0,
        "p75": 0
    })

    if flag:
        print('[')
        flag = False

    # Moved the print_result function call here
    print_result(existing_data)

    if i < len(date_range) - 1:
        print(',')

print(']')
import matplotlib.pyplot as plt

# Execute the pipeline
result_cursor = db.PermanentParkings.aggregate(pipeline)
result_list = list(result_cursor)

# Extract relevant data for plotting
days = [data["_id"] for data in result_list]
avg_values = [data["avg"] for data in result_list]
stdDev_values = [data["stdDev"] for data in result_list]
median_values = [data["median"] for data in result_list]
p25_values = [data["p25"] for data in result_list]
p75_values = [data["p75"] for data in result_list]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(days, avg_values, label='Average')
plt.plot(days, stdDev_values, label='Standard Deviation')
plt.plot(days, median_values, label='Median')
plt.plot(days, p25_values, label='P25')
plt.plot(days, p75_values, label='P75')

plt.xlabel('Day')
plt.ylabel('Value')
plt.title('Vancouver(Parking) stats on December ')
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


#MILAAANNN
def print_result(data):
    print('{\n"Day" : ' + str(data["_id"]) + ',\n"Total Rentals" : ' + str(data["count"]) +
        ',\n"Average" : ' + str(data["avg"]) + ',\n"Std Dev" : ' + str(data["stdDev"]) +
        ',\n"Median" : ' + str(data["median"]) + ',\n"P25" : ' + str(data["p25"]) +
        ',\n"P75" : ' + str(data["p75"]) + '\n}')

# Define start and end dates
start_date = datetime.datetime(2017, 12, 1, 0, 0, 0)
end_date = datetime.datetime(2017, 12, 31, 23, 59, 59)

# Convert dates to Unix time
start_unix_time = int(start_date.timestamp())
end_unix_time = int(end_date.timestamp())

# Generate date range
date_range = [{"day": day} for day in range(1, 32)]  # Assuming 1 to 31 for simplicity

# MongoDB aggregation pipeline
pipeline = [
    {
        "$match": {
            "init_time": {"$gte": start_unix_time, "$lte": end_unix_time},
            "city": "Milano"
        }
    },
    {
        "$project": {
            "_id": 0,
            "city": 1,
            "init_date": 1,
            "duration": {"$ceil": {"$divide": [{"$subtract": ["$final_time", "$init_time"]}, 60]}},
            "moved": {
                "$ne": [
                    {"$arrayElemAt": ["$origin_destination.coordinates", 0]},
                    {"$arrayElemAt": ["$origin_destination.coordinates", 1]}
                ]
            }
        }
    },
    {
        "$match": {
            "duration": {"$gte": 3, "$lte": 150}
        }
    },
    {
        "$project": {
            "city": 1,
            "duration": 1,
            "moved": 1,
            "day": {"$dayOfMonth": "$init_date"}
        }
    },
    {
        "$group": {
            "_id": "$day",
            "count": {"$sum": 1},
            "avg": {"$avg": "$duration"},
            "stdDev": {"$stdDevPop": "$duration"},
            "durations": {"$push": "$duration"}
        }
    },
    {
        "$sort": {"_id": 1}
    },
    {
        "$project": {
            "_id": 1,
            "count": 1,
            "avg": 1,
            "stdDev": 1,
            "median":{"$arrayElemAt":["$durations",
              {"$floor":{ "$multiply": [0.5,{"$size": "$durations"}]}}]},
            "p25": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.25, {"$size": "$durations"}]}}]},
            "p75": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.75, {"$size": "$durations"}]}}]}
        }
    }
]

# Execute the pipeline
result_cursor = db.enjoy_PermanentBookings.aggregate(pipeline)
result_list = list(result_cursor)

# Print the results
flag = True
for i, date in enumerate(date_range):
    existing_data = next((data for data in result_list if data["_id"] == date["day"]), {
        "_id": date["day"],
        "count": 0,
        "avg": 0,
        "stdDev": 0,
        "median": 0,
        "p25": 0,
        "p75": 0
    })

    if flag:
        print('[')
        flag = False

    # Moved the print_result function call here
    print_result(existing_data)

    if i < len(date_range) - 1:
        print(',')

print(']')
import matplotlib.pyplot as plt

# Execute the pipeline
result_cursor = db.enjoy_PermanentBookings.aggregate(pipeline)
result_list = list(result_cursor)

# Extract relevant data for plotting
days = [data["_id"] for data in result_list]
avg_values = [data["avg"] for data in result_list]
stdDev_values = [data["stdDev"] for data in result_list]
median_values = [data["median"] for data in result_list]
p25_values = [data["p25"] for data in result_list]
p75_values = [data["p75"] for data in result_list]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(days, avg_values, label='Average')
plt.plot(days, stdDev_values, label='Standard Deviation')
plt.plot(days, median_values, label='Median')
plt.plot(days, p25_values, label='P25')
plt.plot(days, p75_values, label='P75')

plt.xlabel('Day')
plt.ylabel('Value')
plt.title('Milano(Booking) stats on December ')
plt.legend()
plt.grid(True)
plt.show()


# In[10]:


#MILAAANNN
def print_result(data):
    print('{\n"Day" : ' + str(data["_id"]) + ',\n"Total Rentals" : ' + str(data["count"]) +
        ',\n"Average" : ' + str(data["avg"]) + ',\n"Std Dev" : ' + str(data["stdDev"]) +
        ',\n"Median" : ' + str(data["median"]) + ',\n"P25" : ' + str(data["p25"]) +
        ',\n"P75" : ' + str(data["p75"]) + '\n}')

# Define start and end dates
start_date = datetime.datetime(2017, 12, 1, 0, 0, 0)
end_date = datetime.datetime(2017, 12, 31, 23, 59, 59)

# Convert dates to Unix time
start_unix_time = int(start_date.timestamp())
end_unix_time = int(end_date.timestamp())

# Generate date range
date_range = [{"day": day} for day in range(1, 32)]  # Assuming 1 to 31 for simplicity

# MongoDB aggregation pipeline
pipeline = [
    {
        "$match": {
            "init_time": {"$gte": start_unix_time, "$lte": end_unix_time},
            "city": "Milano"
        }
    },
    {
        "$project": {
            "_id": 0,
            "city": 1,
            "init_date": 1,
            "duration": {"$ceil": {"$divide": [{"$subtract": ["$final_time", "$init_time"]}, 60]}},
            "moved": {
                "$eq": [
                    {"$arrayElemAt": ["$origin_destination.coordinates", 0]},
                    {"$arrayElemAt": ["$origin_destination.coordinates", 1]}
                ]
            }
        }
    },
    {
        "$match": {
            "duration": {"$gte": 3, "$lte": 150}
        }
    },
    {
        "$project": {
            "city": 1,
            "duration": 1,
            "moved": 1,
            "day": {"$dayOfMonth": "$init_date"}
        }
    },
    {
        "$group": {
            "_id": "$day",
            "count": {"$sum": 1},
            "avg": {"$avg": "$duration"},
            "stdDev": {"$stdDevPop": "$duration"},
            "durations": {"$push": "$duration"}
        }
    },
    {
        "$sort": {"_id": 1}
    },
    {
        "$project": {
            "_id": 1,
            "count": 1,
            "avg": 1,
            "stdDev": 1,
            "median":{"$arrayElemAt":["$durations",
              {"$floor":{ "$multiply": [0.5,{"$size": "$durations"}]}}]},
            "p25": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.25, {"$size": "$durations"}]}}]},
            "p75": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.75, {"$size": "$durations"}]}}]}
        }
    }
]

# Execute the pipeline
result_cursor = db.enjoy_PermanentParkings.aggregate(pipeline)
result_list = list(result_cursor)

# Print the results
flag = True
for i, date in enumerate(date_range):
    existing_data = next((data for data in result_list if data["_id"] == date["day"]), {
        "_id": date["day"],
        "count": 0,
        "avg": 0,
        "stdDev": 0,
        "median": 0,
        "p25": 0,
        "p75": 0
    })

    if flag:
        print('[')
        flag = False

    # Moved the print_result function call here
    print_result(existing_data)

    if i < len(date_range) - 1:
        print(',')

print(']')
import matplotlib.pyplot as plt

# Execute the pipeline
result_cursor = db.enjoy_PermanentParkings.aggregate(pipeline)
result_list = list(result_cursor)

# Extract relevant data for plotting
days = [data["_id"] for data in result_list]
avg_values = [data["avg"] for data in result_list]
stdDev_values = [data["stdDev"] for data in result_list]
median_values = [data["median"] for data in result_list]
p25_values = [data["p25"] for data in result_list]
p75_values = [data["p75"] for data in result_list]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(days, avg_values, label='Average')
plt.plot(days, stdDev_values, label='Standard Deviation')
plt.plot(days, median_values, label='Median')
plt.plot(days, p25_values, label='P25')
plt.plot(days, p75_values, label='P75')

plt.xlabel('Day')
plt.ylabel('Value')
plt.title('Milano(Parking) stats on December ')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[11]:


#Romaabooking
def print_result(data):
    print('{\n"Day" : ' + str(data["_id"]) + ',\n"Total Rentals" : ' + str(data["count"]) +
        ',\n"Average" : ' + str(data["avg"]) + ',\n"Std Dev" : ' + str(data["stdDev"]) +
        ',\n"Median" : ' + str(data["median"]) + ',\n"P25" : ' + str(data["p25"]) +
        ',\n"P75" : ' + str(data["p75"]) + '\n}')

# Define start and end dates
start_date = datetime.datetime(2017, 12, 1, 0, 0, 0)
end_date = datetime.datetime(2017, 12, 31, 23, 59, 59)

# Convert dates to Unix time
start_unix_time = int(start_date.timestamp())
end_unix_time = int(end_date.timestamp())

# Generate date range
date_range = [{"day": day} for day in range(1, 32)]  # Assuming 1 to 31 for simplicity

# MongoDB aggregation pipeline
pipeline = [
    {
        "$match": {
            "init_time": {"$gte": start_unix_time, "$lte": end_unix_time},
            "city": "Roma"
        }
    },
    {
        "$project": {
            "_id": 0,
            "city": 1,
            "init_date": 1,
            "duration": {"$ceil": {"$divide": [{"$subtract": ["$final_time", "$init_time"]}, 60]}},
            "moved": {
                "$ne": [
                    {"$arrayElemAt": ["$origin_destination.coordinates", 0]},
                    {"$arrayElemAt": ["$origin_destination.coordinates", 1]}
                ]
            }
        }
    },
    {
        "$match": {
            "duration": {"$gte": 3, "$lte": 150}
        }
    },
    {
        "$project": {
            "city": 1,
            "duration": 1,
            "moved": 1,
            "day": {"$dayOfMonth": "$init_date"}
        }
    },
    {
        "$group": {
            "_id": "$day",
            "count": {"$sum": 1},
            "avg": {"$avg": "$duration"},
            "stdDev": {"$stdDevPop": "$duration"},
            "durations": {"$push": "$duration"}
        }
    },
    {
        "$sort": {"_id": 1}
    },
    {
        "$project": {
            "_id": 1,
            "count": 1,
            "avg": 1,
            "stdDev": 1,
            "median":{"$arrayElemAt":["$durations",
              {"$floor":{ "$multiply": [0.5,{"$size": "$durations"}]}}]},
            "p25": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.25, {"$size": "$durations"}]}}]},
            "p75": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.75, {"$size": "$durations"}]}}]}
        }
    }
]

# Execute the pipeline
result_cursor = db.enjoy_PermanentBookings.aggregate(pipeline)
result_list = list(result_cursor)

# Print the results
flag = True
for i, date in enumerate(date_range):
    existing_data = next((data for data in result_list if data["_id"] == date["day"]), {
        "_id": date["day"],
        "count": 0,
        "avg": 0,
        "stdDev": 0,
        "median": 0,
        "p25": 0,
        "p75": 0
    })

    if flag:
        print('[')
        flag = False

    # Moved the print_result function call here
    print_result(existing_data)

    if i < len(date_range) - 1:
        print(',')

print(']')
import matplotlib.pyplot as plt

# Execute the pipeline
result_cursor = db.enjoy_PermanentBookings.aggregate(pipeline)
result_list = list(result_cursor)

# Extract relevant data for plotting
days = [data["_id"] for data in result_list]
avg_values = [data["avg"] for data in result_list]
stdDev_values = [data["stdDev"] for data in result_list]
median_values = [data["median"] for data in result_list]
p25_values = [data["p25"] for data in result_list]
p75_values = [data["p75"] for data in result_list]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(days, avg_values, label='Average')
plt.plot(days, stdDev_values, label='Standard Deviation')
plt.plot(days, median_values, label='Median')
plt.plot(days, p25_values, label='P25')
plt.plot(days, p75_values, label='P75')

plt.xlabel('Day')
plt.ylabel('Value')
plt.title('Roma(Booking) stats on December ')
plt.legend()
plt.grid(True)
plt.show()


# In[13]:


#Roma
def print_result(data):
    print('{\n"Day" : ' + str(data["_id"]) + ',\n"Total Rentals" : ' + str(data["count"]) +
        ',\n"Average" : ' + str(data["avg"]) + ',\n"Std Dev" : ' + str(data["stdDev"]) +
        ',\n"Median" : ' + str(data["median"]) + ',\n"P25" : ' + str(data["p25"]) +
        ',\n"P75" : ' + str(data["p75"]) + '\n}')

# Define start and end dates
start_date = datetime.datetime(2017, 12, 1, 0, 0, 0)
end_date = datetime.datetime(2017, 12, 31, 23, 59, 59)

# Convert dates to Unix time
start_unix_time = int(start_date.timestamp())
end_unix_time = int(end_date.timestamp())

# Generate date range
date_range = [{"day": day} for day in range(1, 32)]  # Assuming 1 to 31 for simplicity

# MongoDB aggregation pipeline
pipeline = [
    {
        "$match": {
            "init_time": {"$gte": start_unix_time, "$lte": end_unix_time},
            "city": "Roma"
        }
    },
    {
        "$project": {
            "_id": 0,
            "city": 1,
            "init_date": 1,
            "duration": {"$ceil": {"$divide": [{"$subtract": ["$final_time", "$init_time"]}, 60]}},
            "moved": {
                "$eq": [
                    {"$arrayElemAt": ["$origin_destination.coordinates", 0]},
                    {"$arrayElemAt": ["$origin_destination.coordinates", 1]}
                ]
            }
        }
    },
    {
        "$match": {
            "duration": {"$gte": 3, "$lte": 150}
        }
    },
    {
        "$project": {
            "city": 1,
            "duration": 1,
            "moved": 1,
            "day": {"$dayOfMonth": "$init_date"}
        }
    },
    {
        "$group": {
            "_id": "$day",
            "count": {"$sum": 1},
            "avg": {"$avg": "$duration"},
            "stdDev": {"$stdDevPop": "$duration"},
            "durations": {"$push": "$duration"}
        }
    },
    {
        "$sort": {"_id": 1}
    },
    {
        "$project": {
            "_id": 1,
            "count": 1,
            "avg": 1,
            "stdDev": 1,
            "median":{"$arrayElemAt":["$durations",
              {"$floor":{ "$multiply": [0.5,{"$size": "$durations"}]}}]},
            "p25": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.25, {"$size": "$durations"}]}}]},
            "p75": {"$arrayElemAt": ["$durations", {"$floor": {"$multiply": [0.75, {"$size": "$durations"}]}}]}
        }
    }
]

# Execute the pipeline
result_cursor = db.enjoy_PermanentParkings.aggregate(pipeline)
result_list = list(result_cursor)

# Print the results
flag = True
for i, date in enumerate(date_range):
    existing_data = next((data for data in result_list if data["_id"] == date["day"]), {
        "_id": date["day"],
        "count": 0,
        "avg": 0,
        "stdDev": 0,
        "median": 0,
        "p25": 0,
        "p75": 0
    })

    if flag:
        print('[')
        flag = False

    # Moved the print_result function call here
    print_result(existing_data)

    if i < len(date_range) - 1:
        print(',')

print(']')
import matplotlib.pyplot as plt

# Execute the pipeline
result_cursor = db.enjoy_PermanentParkings.aggregate(pipeline)
result_list = list(result_cursor)

# Extract relevant data for plotting
days = [data["_id"] for data in result_list]
avg_values = [data["avg"] for data in result_list]
stdDev_values = [data["stdDev"] for data in result_list]
median_values = [data["median"] for data in result_list]
p25_values = [data["p25"] for data in result_list]
p75_values = [data["p75"] for data in result_list]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(days, avg_values, label='Average')
plt.plot(days, stdDev_values, label='Standard Deviation')
plt.plot(days, median_values, label='Median')
plt.plot(days, p25_values, label='P25')
plt.plot(days, p75_values, label='P75')

plt.xlabel('Day')
plt.ylabel('Value')
plt.title('Roma(Parking) stats on December ')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




