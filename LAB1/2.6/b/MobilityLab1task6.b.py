#!/usr/bin/env python
# coding: utf-8

# In[28]:


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

tzVancouver=-8;  #Time zone shift for
time_week = 7*24*60*60
init_nov_c = '2017-12-01 00:00:00'
final_nov_c = '2017-12-01 23:59:59'
init_nov = (time.mktime(datetime.datetime.strptime(init_nov_c,'%Y-%m-%d %H:%M:%S').timetuple()))
final_nov = (time.mktime(datetime.datetime.strptime(final_nov_c,'%Y-%m-%d %H:%M:%S').timetuple()))

recorded = PP_C2.find({"$and":[{"city":'Vancouver'}, 
                               {"init_time":{"$gte":init_nov+tzVancouver*60*60}},
                               {"init_time":{"$lte":final_nov+tzVancouver*60*60}}]}, 
                      {'_id':1, 'city':1,'loc':1})
test = pd.DataFrame(recorded)
resultado = [d.get('coordinates') for d in test['loc']]
parking = np.zeros((len(test),2))
for i in range (len(test)):
    parking[i,1] = resultado[i][0]
    parking[i,0] = resultado[i][1]
    
parking_coordinates = pd.DataFrame(parking)
parking_coordinates.rename(columns={0:"latitude", 1:"longitude"}, inplace=True)
parking_coordinates.to_csv (r'C:\Users\User\parkinggg.csv', index = False, header=True)

df = pd.read_csv("parkinggg.csv") 
geometry1 = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]  

geo_df1 = gpd.GeoDataFrame(df, geometry=geometry1) 

Vancouver_map = gpd.read_file(r'E:\ict\mobility\local-area-boundary.shp')
Vancouver_map.to_crs(epsg = 4326 , inplace = True)
c=0
counter=[]
for i in Vancouver_map['geometry']:
    for j in geo_df1['geometry']:
        if i.contains(j):
            c+=1
    counter.append(c)
    c=0
Vancouver_map['counter']=counter

Vancouver_map.plot(column = 'counter',cmap='viridis',legend = True,figsize=(10,10))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('vancouver Parkings First of December 2017')


# In[23]:



tzVancouver=-8;  #Time zone shift for
time_week = 7*24*60*60
init_nov_c = '2018-01-01 00:00:00'
final_nov_c = '2018-01-01 23:59:59'
init_nov = (time.mktime(datetime.datetime.strptime(init_nov_c,'%Y-%m-%d %H:%M:%S').timetuple()))
final_nov = (time.mktime(datetime.datetime.strptime(final_nov_c,'%Y-%m-%d %H:%M:%S').timetuple()))

recorded = PP_C2.find({"$and":[{"city":'Vancouver'}, 
                               {"init_time":{"$gte":init_nov+tzVancouver*60*60}},
                               {"init_time":{"$lte":final_nov+tzVancouver*60*60}}]}, 
                      {'_id':1, 'city':1,'loc':1})
test = pd.DataFrame(recorded)
resultado = [d.get('coordinates') for d in test['loc']]
parking = np.zeros((len(test),2))
for i in range (len(test)):
    parking[i,1] = resultado[i][0]
    parking[i,0] = resultado[i][1]
    
parking_coordinates = pd.DataFrame(parking)
parking_coordinates.rename(columns={0:"latitude", 1:"longitude"}, inplace=True)
parking_coordinates.to_csv (r'C:\Users\User\parkingg.csv', index = False, header=True)

df = pd.read_csv("parkingg.csv") 
geometry1 = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]  

geo_df1 = gpd.GeoDataFrame(df, geometry=geometry1)

Vancouver_map = gpd.read_file(r'E:\ict\mobility\local-area-boundary.shp')
Vancouver_map.to_crs(epsg = 4326 , inplace = True)
c=0
counter=[]
for i in Vancouver_map['geometry']:
    for j in geo_df1['geometry']:
        if i.contains(j):
            c+=1
    counter.append(c)
    c=0
Vancouver_map['counter']=counter

Vancouver_map.plot(column = 'counter',cmap='viridis',legend = True,figsize=(10,10))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('vancouver Parkings First of January 2018')


# In[25]:


tzVancouver=-8;  #Time zone shift for
time_week = 7*24*60*60
init_nov_c = '2017-11-01 00:00:00'
final_nov_c = '2017-11-01 23:59:59'
init_nov = (time.mktime(datetime.datetime.strptime(init_nov_c,'%Y-%m-%d %H:%M:%S').timetuple()))
final_nov = (time.mktime(datetime.datetime.strptime(final_nov_c,'%Y-%m-%d %H:%M:%S').timetuple()))

recorded = PP_C2.find({"$and":[{"city":'Vancouver'}, 
                               {"init_time":{"$gte":init_nov+tzVancouver*60*60}},
                               {"init_time":{"$lte":final_nov+tzVancouver*60*60}}]}, 
                      {'_id':1, 'city':1,'loc':1})
test = pd.DataFrame(recorded)
resultado = [d.get('coordinates') for d in test['loc']]
parking = np.zeros((len(test),2))
for i in range (len(test)):
    parking[i,1] = resultado[i][0]
    parking[i,0] = resultado[i][1]
    
parking_coordinates = pd.DataFrame(parking)
parking_coordinates.rename(columns={0:"latitude", 1:"longitude"}, inplace=True)
parking_coordinates.to_csv (r'C:\Users\User\parkingg.csv', index = False, header=True)

df = pd.read_csv("parkingg.csv") 
geometry1 = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]  

geo_df1 = gpd.GeoDataFrame(df, geometry=geometry1)

Vancouver_map = gpd.read_file(r'E:\ict\mobility\local-area-boundary.shp')
Vancouver_map.to_crs(epsg = 4326 , inplace = True)
c=0
counter=[]
for i in Vancouver_map['geometry']:
    for j in geo_df1['geometry']:
        if i.contains(j):
            c+=1
    counter.append(c)
    c=0
Vancouver_map['counter']=counter

Vancouver_map.plot(column = 'counter',cmap='viridis',legend = True,figsize=(10,10))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('vancouver Parkings First of November 2017')


# In[ ]:




