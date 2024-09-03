#!/usr/bin/env python
# coding: utf-8

# In[19]:


import matplotlib.pyplot as plt
import numpy as np

Vancouver_permanentB = [10557,10562,6500,3907,5724,8715,16424,27164,27097,25662,23637,25037,26019,26219,28951,33815,39887,41111,39051,34896,28257,25782,20784,15520]
Vancouver_permanentP = [12442,11624,6877,4264,4466,8385,12175,21859,26787,27578,23678,23862,25080,25895,27135,31739,36600,40750,41411,39583,31510,26788,23324,17600
                           ]


Milano_permanentB = [23941,15686,10056,7237,8567,8961,10252,15650,24236,25353,23211,24849,28528,28516,29917,30546,31705,34728,38656,39857,37940,33618,31380,29260 ]
Milano_permanentP = [26790,20568,13024,8460,7995,8340,8576,11538,19896,27468,24259,25169,27013,29324,29968,29393,30675,32297,35349,39167,40842,37224,31740,29905 ]



Roma_permanentB = [14586,9525,5636,3463,4352,5457,6987,13690,17738,18145,16950,17615,19901,20115,19532,21666,22806,23458,26266,26373,25025,19485,18137,17990]

Roma_permanentP = [16590,12110,7456,4388,4380,6475,5187,10059,13938,18975,17220,18004,18303,20597,20218,20164,21646,22553,24818,27799,26935,23215,17622,17977]

# Bookings Aggregated per Hour
fig, ax1 = plt.subplots()
ax1.plot(h, Vancouver_permanentB, 's-', label='Vancouver', color='blue', markersize=8, linewidth=2)
ax1.plot(h, Milano_permanentB, 'o--', label='Milano', color='green', markersize=8, linewidth=2)
ax1.plot(h, Roma_permanentB, 'D-.', label='Roma', color='red', markersize=8, linewidth=2)
ax1.legend()
ax1.set_title('Booking Aggregation', fontsize=16)
ax1.grid(linewidth=0.5)
ax1.set_xticks(h)
ax1.set_xlabel('Hours of the day', fontsize=14)
ax1.set_ylabel('Booked cars', fontsize=14)

# Parkings Aggregated per Hour
fig, ax2 = plt.subplots()
ax2.plot(h, Vancouver_permanentP, 's-', label='Vancouver', color='blue', markersize=8, linewidth=2)
ax2.plot(h, Milano_permanentP, 'o--', label='Milano', color='green', markersize=8, linewidth=2)
ax2.plot(h, Roma_permanentP, 'D-.', label='Roma', color='red', markersize=8, linewidth=2)
ax2.legend()
ax2.set_title('Parking Aggregation', fontsize=16)
ax2.grid(linewidth=0.5)
ax2.set_xticks(h)
ax2.set_xlabel('Hours of the day', fontsize=14)
ax2.set_ylabel('Parked cars', fontsize=14)

plt.show()


# In[21]:


#task4
import matplotlib.pyplot as plt
import numpy as np

Vancouver_permanentB = [6623,3979,2637,1738,2100,5159,10589,16674,17740,16507,14917,16343,17420,18065,19777,23383,27424,28226,26878,23453,19286,16967,14111,10162]
Vancouver_permanentP = [11661,11117,6561,4070,4279,8031,10843,18016,21095,23220,20804,20776,21679,22535,23497,26669,28741,30147,30769,32013,27330,23791,21131,16252]


Milano_permanentB = [21908,14380,9184,6626,7312,7715,9713,14642,22524,23120,21053,22718,26434,26371,27816,28408,29469,32158,35264,37043,35448,31458,29498,27323]
Milano_permanentP = [ 20833,16963,10987,7369,7210,7920,8295,10861,17477,24034,21929,22808,23914,25922,26782,25914,26871,27716,29100,30411,30946,29905,25777,23410]



Roma_permanentB = [13400,8759,5145,3097,4100,3681,6584,11995,16342,16453,15275,15906,18264,18555,18020,20071,21086,21701,22908,24546,23271,18225,17004,16855]

Roma_permanentP = [14244,10819,6775,4062,4192,5489,5063,9616,12827,17519,15889,16634,16680,18825,18670,18284,19289,19953,21648,23794,22685,20436,15453,15395]

# Bookings Aggregated per Hour
fig, ax1 = plt.subplots()
ax1.plot(h, Vancouver_permanentB, 's-', label='Vancouver', color='blue', markersize=8, linewidth=2)
ax1.plot(h, Milano_permanentB, 'o--', label='Milano', color='green', markersize=8, linewidth=2)
ax1.plot(h, Roma_permanentB, 'D-.', label='Roma', color='red', markersize=8, linewidth=2)
ax1.legend()
ax1.set_title('Filtered_Booking Aggregation', fontsize=16)
ax1.grid(linewidth=0.5)
ax1.set_xticks(h)
ax1.set_xlabel('Hours of the day', fontsize=14)
ax1.set_ylabel('Booked cars', fontsize=14)

# Parkings Aggregated per Hour
fig, ax2 = plt.subplots()
ax2.plot(h, Vancouver_permanentP, 's-', label='Vancouver', color='blue', markersize=8, linewidth=2)
ax2.plot(h, Milano_permanentP, 'o--', label='Milano', color='green', markersize=8, linewidth=2)
ax2.plot(h, Roma_permanentP, 'D-.', label='Roma', color='red', markersize=8, linewidth=2)
ax2.legend()
ax2.set_title('Filtered_Parking Aggregation', fontsize=16)
ax2.grid(linewidth=0.5)
ax2.set_xticks(h)
ax2.set_xlabel('Hours of the day', fontsize=14)
ax2.set_ylabel('Parked cars', fontsize=14)

plt.show()


# In[ ]:




