#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pprint
import datetime 
import time
import warnings
import pymongo as pm #import MongoClient only
import os
import pandas as pd
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
from statsmodels.tsa.arima_model import ARIMA 
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import acf, pacf
import seaborn as sn
import csv 
import numpy as np
client = pm.MongoClient('bigdatadb.polito.it',
 ssl=True,
 authSource = 'carsharing',
 username = 'ictts',
 password ='Ict4SM22!',
 tlsAllowInvalidCertificates=True)
db = client['carsharing']
PB_C2 = db['PermanentBookings']
AB_C2 = db['ActiveBookings']
PP_C2 = db['PermanentParkings']
AP_C2 = db['ActiveParkings']
PB_EJ = db['enjoy_PermanentBookings']
AB_EJ = db['enjoy_ActiveBookings']
PP_EJ = db['enjoy_PermanentParkings']
AP_EJ = db['enjoyActiveParkings']

cities = ["Vancouver", "Milano", "Roma"]


# In[3]:


city = "Milano"
start_date = datetime.datetime(2018, 1, 1, 0, 0, 0)
end_date = datetime.datetime(2018, 1, 30, 23, 59, 59)

# Convert dates to Unix timestamps
init_jan = int(start_date.timestamp())
final_jan = int(end_date.timestamp())

# Aggregation pipeline
bookings = PB_EJ.aggregate([
    {
        "$match": {
            "$and": [
                {"city": city},
                {"init_time": {"$gte": init_jan}},
                {"init_time": {"$lte": final_jan}}
            ]
        }
    },
    {
        "$project": {
            "_id": 0,
            "moved": {
                "$ne": [
                    {"$arrayElemAt": ["$origin_destination.coordinates", 0]},
                    {"$arrayElemAt": ["$origin_destination.coordinates", 1]}
                ]
            },
            "bookingDuration": {"$subtract": ["$final_time", "$init_time"]},
            "init_time": 1
        }
    },
    {
        "$match": {
            "$and": [
                {"moved": True},
                {"bookingDuration": {"$gte": 180}},
                {"bookingDuration": {"$lte": 9000}}
            ]
        }
    }
])


# In[4]:


# Convert the query result to a list
temp = list(bookings)

# Sort bookings by initial time
sortedBookings = sorted(temp, key=lambda i: i['init_time'])

# Helper function to convert timestamp to hour of the month
def timestamp_to_hour(timestamp, start_timestamp):
    return int((timestamp - start_timestamp) / 3600)

# Count bookings per hour
hourly_counts = {}
for booking in sortedBookings:
    hour_of_month = timestamp_to_hour(booking['init_time'], init_jan)
    hourly_counts[hour_of_month] = hourly_counts.get(hour_of_month, 0) + 1

# Writing to CSV
with open(f'Rentals_{city}_hourly.csv', 'w') as outfile:
    fields = ['hour_of_month', 'numberOfbookings']
    write = csv.DictWriter(outfile, fieldnames=fields)
    write.writeheader()
    for hour, count in hourly_counts.items():
        write.writerow({'hour_of_month': hour, 'numberOfbookings': count})

# Plotting
fig = plt.figure(1, figsize=(12, 6))
hours = list(hourly_counts.keys())
counts = list(hourly_counts.values())
plt.plot(hours, counts, label=city)
plt.title(f"Number of bookings per hour in January in {city}")
plt.xlabel("Hour of the Month")
plt.ylabel("Number of Bookings")
plt.grid(True)
plt.legend()
plt.show()


# In[5]:


###check for missing data task2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
# Convert hourly_counts to a DataFrame.
df = pd.DataFrame(list(hourly_counts.items()), columns=['Time', 'rental'])

# Ensure 'Time' is treated as an integer.
df['Time'] = pd.to_numeric(df['Time'], downcast='integer', errors='coerce').fillna(0).astype(int)
df = df.set_index('Time')

# Initialize a new DataFrame to fill missing hours with zeros.
df2 = pd.DataFrame(columns=['Time', 'rental'])

# Safety check: Proceed only if df is not empty and does not contain NaN in the index.
if not df.empty and not df.index.isna().any():
    min_hour = df.index.min()
    max_hour = df.index.max()

    # Iterate over the expected range of hours and check for missing entries.
    for i in range(min_hour, max_hour + 1):
        if i not in df.index:
            df2 = df2.append({'Time': i, 'rental': 0}, ignore_index=True)

    # Prepare df2 for merging.
    df2['Time'] = pd.to_numeric(df2['Time'], downcast='integer')
    df2 = df2.set_index('Time')

    # Combine the original data with the filled-in missing data and sort.
    df_combined = pd.concat([df, df2]).sort_index()
    df_combined['rental'] = pd.to_numeric(df_combined['rental'], errors='coerce')

    # Plot the data.
    df_combined.plot(figsize=(15,5), title=f"Number of bookings per hour in January after checking for missing data in {city}")
    plt.xlabel("Hour of the Month")
    plt.ylabel("Number of Bookings")
    plt.grid(True)
    plt.show()

    # Write the combined data to CSV.
    with open(f'Complete_Rentals_{city}_hourly.csv', 'w') as outfile:
        df_combined.to_csv(outfile, header=True)
else:
    print("The DataFrame is empty or the index contains NaN values. Please check your data.")


# In[6]:


###task3 it is stationary so assume that d=0
df = pd.DataFrame(list(hourly_counts.items()), columns=['Hour', 'Bookings'])

# Set the 'Hour' column as the index
df.set_index('Hour', inplace=True)

# Calculate the rolling mean and standard deviation with a 1 week window
rolling_mean = df['Bookings'].rolling(window=168).mean()
rolling_std = df['Bookings'].rolling(window=168).std()

# Plotting the original time series, rolling mean, and rolling standard deviation
plt.figure(figsize=(12, 6))
plt.plot(df['Bookings'], color='blue', label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_std, color='black', label='Rolling Std ')
plt.legend(loc='best')
plt.title(' Mean & Standard Deviation for Milano enjoy window size = 1 week')
plt.xlabel('Hours')
plt.ylabel('Number of Bookings')
plt.show()


# In[7]:


###task4 acf for all sample
###correlation in daily times,also in same say and also weakly baisis so there is priodicity
df = pd.DataFrame(list(hourly_counts.items()), columns=['Hour', 'Bookings'])

# Set the 'Hour' column as the index
df.set_index('Hour', inplace=True)

# Compute ACF and PACF
lag_acf = acf(df['Bookings'], nlags=744)
#lag_pacf = pacf(df['Bookings'], nlags=48, method='ols')

# Plotting ACF using stem plot
plt.figure(figsize=(12, 6))
plt.subplot(121) 
plt.stem(lag_acf, use_line_collection=True)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df['Bookings'])), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df['Bookings'])), linestyle='--', color='gray')
plt.title('Autocorrelation Function (ACF) Milano ')
plt.xlabel('Lags')
plt.ylabel('ACF')


# In[8]:


###task4 acf for 48 lags zooming
###as the ACF better corrlation in 24h and 48h not negligible 
##the PacF is quickly drops to almost negligible so it maybe is AR model
df = pd.DataFrame(list(hourly_counts.items()), columns=['Hour', 'Bookings'])

# Set the 'Hour' column as the index
df.set_index('Hour', inplace=True)

# Compute ACF and PACF
lag_acf = acf(df['Bookings'], nlags=48)
lag_pacf = pacf(df['Bookings'], nlags=48, method='ols')

# Plotting ACF using stem plot
plt.figure(figsize=(12, 6))
plt.subplot(121) 
plt.stem(lag_acf, use_line_collection=True)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df['Bookings'])), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df['Bookings'])), linestyle='--', color='gray')
plt.title('Autocorrelation Function (ACF) Milano ')
plt.xlabel('Lags')
plt.ylabel('ACF')

# Plotting PACF using stem plot
plt.subplot(122)
plt.stem(lag_pacf, use_line_collection=True)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df['Bookings'])), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df['Bookings'])), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function (PACF)  Milano')
plt.xlabel('Lags')
plt.ylabel('PACF')
plt.tight_layout()
plt.show()


# In[9]:


from statsmodels.tsa.arima.model import ARIMA
p=2; d=0;q=0
#fit model
model= ARIMA(df.astype(float), order=(p,d,q))#time series that i convert to float and creat a model
model_fit=model.fit()
fig=plt.figure(figsize=(15,5))
plt.plot(df)
plt.plot(model_fit.fittedvalues , color='purple')
plt.title(' prediction over all selected data set for Milano with AR model p equal 2')


# In[10]:


model_fit.summary()


# In[29]:


#task5,6 7.a assuming fixed train and test set and changing p and q and calculating the error metrics
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming hourly_counts is a dictionary with hours as keys and booking counts as values
df = pd.DataFrame(list(hourly_counts.items()), columns=['Hour', 'Bookings'])
df.set_index('Hour', inplace=True)

# Splitting the data into training and test sets
train = df.iloc[:504]  # For example, first 504 hours for training
test = df.iloc[504:552]  # Following hours for testing

# Initialize a dictionary to store the metrics
metrics = {}

# Loop over different values of p and q
for p in range(1, 6):
    for q in range(0, 5):
        try:
            # Fit the ARIMA model
            model = ARIMA(train, order=(p,0,q))
            fitted_model = model.fit(method='innovations_mle')

            # Make predictions
            predictions = fitted_model.predict(start=test.index[0], end=test.index[-1])
            predictions_series = pd.Series(predictions, index=test.index)

            # Calculate metrics
            mae = mean_absolute_error(test['Bookings'], predictions_series)
            mse = mean_squared_error(test['Bookings'], predictions_series)
            r2 = r2_score(test['Bookings'], predictions_series)
            mape = np.mean(np.abs((test['Bookings'] - predictions_series) / test['Bookings'])) * 100

            # Store metrics
            metrics[(p, q)] = {'MAE': mae, 'MSE': mse, 'R2': r2, 'MAPE': mape}
        except Exception as e:
            print(f"Failed to fit ARIMA({p},0,{q}): {e}")

# Print metrics for each (p, q) combination
for (p, q), vals in metrics.items():
    print(f"ARIMA({p},0,{q}) - MAE: {vals['MAE']:.2f}, MSE: {vals['MSE']:.2f}, R2: {vals['R2']:.2f}, MAPE: {vals['MAPE']:.2f}%")
# Extract metrics for plotting
mae_values = np.zeros((5, 5))
mse_values = np.zeros((5, 5))
r2_values = np.zeros((5, 5))
mape_values = np.zeros((5, 5))

for i, p in enumerate(range(1, 6)):
    for j, q in enumerate(range(0, 5)):
        if (p, q) in metrics:
            mae_values[i, j] = metrics[(p, q)]['MAE']
            mse_values[i, j] = metrics[(p, q)]['MSE']
            r2_values[i, j] = metrics[(p, q)]['R2']
            mape_values[i, j] = metrics[(p, q)]['MAPE']

# Plotting heatmap for MAE
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(mae_values, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('MAE Heatmap Milano')
plt.xlabel('q values')
plt.ylabel('p values')
plt.xticks(np.arange(5), np.arange(5))
plt.yticks(np.arange(5), np.arange(1, 6))

# Plotting heatmap for MSE
plt.subplot(2, 2, 2)
plt.imshow(mse_values, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('MSE Heatmap Milano')
plt.xlabel('q values')
plt.ylabel('p values')
plt.xticks(np.arange(5), np.arange(5))
plt.yticks(np.arange(5), np.arange(1, 6))

# Plotting heatmap for R2
plt.subplot(2, 2, 3)
plt.imshow(r2_values, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('R2 Heatmap Milano')
plt.xlabel('q values')
plt.ylabel('p values')
plt.xticks(np.arange(5), np.arange(5))
plt.yticks(np.arange(5), np.arange(1, 6))

# Plotting heatmap for MAPE
plt.subplot(2, 2, 4)
plt.imshow(mape_values, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('MAPE Heatmap Milano')
plt.xlabel('q values')
plt.ylabel('p values')
plt.xticks(np.arange(5), np.arange(5))
plt.yticks(np.arange(5), np.arange(1, 6))

plt.tight_layout()
plt.show()


# In[32]:


#sliding window
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Assuming hourly_counts is a dictionary with hours as keys and booking counts as values
df = pd.DataFrame(list(hourly_counts.items()), columns=['Hour', 'Bookings'])
df.set_index('Hour', inplace=True)

# Define the start of the test set
test_start = 504

# Initialize a list to store results
results = []

# Window size for sliding window
window_size = 24  # Assuming each window represents one day (24 hours)

# Fix values for ARIMA model
p_fixed = 4
d_fixed = 0
q_fixed = 3

# Initialize lists to store plotting data
end_train_values = []
mae_values = []
mape_values = []
r2_values = []

# Sliding the window one day at a time
for end_train in range(initial_train_size, test_start, 24):  # Assuming each step slides one day (24 hours)
    train = df.iloc[end_train - window_size:end_train]
    test = df.iloc[end_train:end_train+24]  # Next day for testing
    
    try:
        # Fit the ARIMA model with fixed values
        model = ARIMA(train, order=(p_fixed, d_fixed, q_fixed))
        fitted_model = model.fit()
        
        # Make predictions for the next day
        predictions = fitted_model.predict(start=train.index[-1] + 1, end=train.index[-1] + len(test))
        predictions_series = pd.Series(predictions, index=test.index)
        
        # Calculate metrics
        mae = mean_absolute_error(test['Bookings'], predictions_series)
        mape = np.mean(np.abs((test['Bookings'] - predictions_series) / test['Bookings'])) * 100
        r2 = r2_score(test['Bookings'], predictions_series)
        
        # Store metrics
        results.append({
            'end_train': end_train,
            'p': p_fixed,
            'd': d_fixed,
            'q': q_fixed,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })
        
        # Store data for plotting
        end_train_values.append(end_train)
        mae_values.append(mae)
        mape_values.append(mape)
        r2_values.append(r2)
    except Exception as e:
        print(f"Failed to fit ARIMA({p_fixed},{d_fixed},{q_fixed}): {e}")

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)
# Print the results
print(results_df)

# Plotting section
plt.figure(figsize=(12, 8))

# Plot MAE
plt.subplot(3, 1, 1)
plt.plot(end_train_values, mae_values, marker='o', linestyle='-', color='blue')
plt.title('Mean Absolute Error (MAE) Over Sliding Window')
plt.xlabel('End of Training Set Index')
plt.ylabel('MAE')

# Plot MAPE
plt.subplot(3, 1, 2)
plt.plot(end_train_values, mape_values, marker='o', linestyle='-', color='orange')
plt.title('Mean Absolute Percentage Error (MAPE) Over Sliding Window')
plt.xlabel('End of Training Set Index')
plt.ylabel('MAPE')

# Plot R2
plt.subplot(3, 1, 3)
plt.plot(end_train_values, r2_values, marker='o', linestyle='-', color='green')
plt.title('R-squared (R2) Over Sliding Window')
plt.xlabel('End of Training Set Index')
plt.ylabel('R2')

plt.tight_layout()
plt.show()


# This section assumes you have lists or arrays of actual and predicted values for each window.
# Since the implementation details of all_actual_values and all_predicted_values are not provided,
# here's a conceptual fix:

# Correcting the plotting section for actual vs predicted values
# Assuming each set of actual and predicted values correspond to a window of 24 hours
# The following code block is conceptual and needs to be adjusted according to how you store and access your windowed data
all_actual_values = []
all_predicted_values = []

plt.figure(figsize=(12, 8))

# Example loop for plotting, adjust according to your data structure
for i, end_train in enumerate(end_train_values):
    # Determine the time indices for this window
    time_indices = df.index[end_train-24:end_train]
    
    # Plot actual vs predicted for this window
    # Ensure all_actual_values[i] and all_predicted_values[i] are defined and correspond to this window
    plt.plot(time_indices, all_actual_values[i], label=f"Actual (Window {i + 1})", alpha=0.5)
    plt.plot(time_indices, all_predicted_values[i], label=f"Predicted (Window {i + 1})", linestyle='dashed', alpha=0.7)

plt.title('Actual vs Predicted Values')
plt.xlabel('Hour')
plt.ylabel('Bookings')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:





# In[34]:


#####task 7.b expanding window 
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Assuming hourly_counts is a dictionary with hours as keys and booking counts as values
df = pd.DataFrame(list(hourly_counts.items()), columns=['Hour', 'Bookings'])
df.set_index('Hour', inplace=True)

# Define the start of the test set
test_start = 504

# Initialize a list to store results
results = []

# Initial training set size
initial_train_size = 336  # Start with the first week for training

# Fix values for ARIMA model
p_fixed = 4
d_fixed = 0
q_fixed = 3

# Initialize lists to store plotting data
end_train_values = []
mae_values = []
mape_values = []
r2_values = []

# Initialize lists for actual and predicted values
all_actual_values = []
all_predicted_values = []

# Expanding the window one day at a time
for end_train in range(initial_train_size, test_start, 24):  # Assuming each step adds one day (24 hours)
    train = df.iloc[:end_train]
    test = df.iloc[end_train:end_train+24]  # Next day for testing
    
    try:
        # Fit the ARIMA model with fixed values
        model = ARIMA(train, order=(p_fixed, d_fixed, q_fixed))
        fitted_model = model.fit()
        
        # Make predictions for the entire dataset
        predictions = fitted_model.predict(start=train.index[0], end=df.index[-1])
        
        # Store actual and predicted values
        all_actual_values.append(df['Bookings'])
        all_predicted_values.append(predictions)
        
        # Make predictions for the next day
        predictions_test = fitted_model.predict(start=train.index[-1] + 1, end=train.index[-1] + len(test))
        predictions_series = pd.Series(predictions_test, index=test.index)
        
        # Calculate metrics
        mae = mean_absolute_error(test['Bookings'], predictions_series)
        mape = np.mean(np.abs((test['Bookings'] - predictions_series) / test['Bookings'])) * 100
        r2 = r2_score(test['Bookings'], predictions_series)
        
        # Store metrics
        results.append({
            'end_train': end_train,
            'p': p_fixed,
            'd': d_fixed,
            'q': q_fixed,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })
        
        # Store data for plotting
        end_train_values.append(end_train)
        mae_values.append(mae)
        mape_values.append(mape)
        r2_values.append(r2)
    except Exception as e:
        print(f"Failed to fit ARIMA({p_fixed},{d_fixed},{q_fixed}): {e}")

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)
# Print the results
print(results_df)

# Plotting section
plt.figure(figsize=(12, 16))

# Plot MAE, MAPE, and R2
plt.subplot(4, 1, 1)
plt.plot(end_train_values, mae_values, marker='o', linestyle='-', color='blue', label='MAE')
plt.title('Mean Absolute Error (MAE) Over Expanding Window')
plt.xlabel('End of Training Set Index')
plt.ylabel('MAE')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(end_train_values, mape_values, marker='o', linestyle='-', color='orange', label='MAPE')
plt.title('Mean Absolute Percentage Error (MAPE) Over Expanding Window')
plt.xlabel('End of Training Set Index')
plt.ylabel('MAPE')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(end_train_values, r2_values, marker='o', linestyle='-', color='green', label='R2')
plt.title('R-squared (R2) Over Expanding Window')
plt.xlabel('End of Training Set Index')
plt.ylabel('R2')
plt.legend()

# Plot Actual and Predicted Values
plt.subplot(4, 1, 4)
for i in range(len(all_actual_values)):
    plt.plot(df.index, all_actual_values[i], label=f"Actual (Window {i + 1})", alpha=0.5)
    plt.plot(df.index, all_predicted_values[i], label=f"Predicted (Window {i + 1})", linestyle='dashed', alpha=0.7)

plt.title('Actual vs Predicted Values')
plt.xlabel('Hour')
plt.ylabel('Bookings')
plt.legend()

plt.tight_layout()
plt.show()


# In[20]:


#####error metrics over total period
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Assuming hourly_counts is a dictionary with hours as keys and booking counts as values
df = pd.DataFrame(list(hourly_counts.items()), columns=['Hour', 'Bookings'])
df.set_index('Hour', inplace=True)

# Define the start of the test set
test_start = 504

# Initialize lists to store the true and predicted values for the whole period
true_values = []
predicted_values = []

# Initial training set size
initial_train_size = 336  # Start with the first week for training

# Fix values for ARIMA model
p_fixed = 4
d_fixed = 0
q_fixed = 3

# Expanding the window one day at a time
for end_train in range(initial_train_size, test_start, 24):  # Assuming each step adds one day (24 hours)
    train = df.iloc[:end_train]
    test = df.iloc[end_train:end_train+24]  # Next day for testing
    
    try:
        # Fit the ARIMA model with fixed values
        model = ARIMA(train, order=(p_fixed, d_fixed, q_fixed))
        fitted_model = model.fit()
        
        # Make predictions for the next day
        predictions = fitted_model.predict(start=train.index[-1] + 1, end=train.index[-1] + len(test))
        
        # Append the true values and predictions to the lists
        true_values.extend(test['Bookings'].values)
        predicted_values.extend(predictions)
        
    except Exception as e:
        print(f"Failed to fit ARIMA({p_fixed},{d_fixed},{q_fixed}): {e}")

# Now calculate the metrics over the entire period
mae = mean_absolute_error(true_values, predicted_values)
mape = np.mean(np.abs((np.array(true_values) - np.array(predicted_values)) / np.array(true_values))) * 100
r2 = r2_score(true_values, predicted_values)

# Print the aggregated metrics
print(f"MAE over entire period with expanding window Milano: {mae}")
print(f"MAPE over entire period with expanding window Milano: {mape}")
print(f"R^2 over entire period with expanding window Milano: {r2}")


# In[ ]:





# In[35]:


#####task 7.b expanding window using mle fitmodel
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Assuming hourly_counts is a dictionary with hours as keys and booking counts as values
df = pd.DataFrame(list(hourly_counts.items()), columns=['Hour', 'Bookings'])
df.set_index('Hour', inplace=True)

# Define the start of the test set
test_start = 504

# Initialize a list to store results
results = []

# Initial training set size
initial_train_size = 336  # Start with the first week for training

# Fix values for ARIMA model
p_fixed = 4
d_fixed = 0
q_fixed = 3

# Initialize lists to store plotting data
end_train_values = []
mae_values = []
mape_values = []
r2_values = []

# Initialize lists for actual and predicted values
all_actual_values = []
all_predicted_values = []

# Expanding the window one day at a time
for end_train in range(initial_train_size, test_start, 24):  # Assuming each step adds one day (24 hours)
    train = df.iloc[:end_train]
    test = df.iloc[end_train:end_train+24]  # Next day for testing
    
    try:
        # Fit the ARIMA model with fixed values
        model = ARIMA(train, order=(p_fixed, d_fixed, q_fixed))
        fitted_model = model.fit(method='innovations_mle')
        
        # Make predictions for the entire dataset
        predictions = fitted_model.predict(start=train.index[0], end=df.index[-1])
        
        # Store actual and predicted values
        all_actual_values.append(df['Bookings'])
        all_predicted_values.append(predictions)
        
        # Make predictions for the next day
        predictions_test = fitted_model.predict(start=train.index[-1] + 1, end=train.index[-1] + len(test))
        predictions_series = pd.Series(predictions_test, index=test.index)
        
        # Calculate metrics
        mae = mean_absolute_error(test['Bookings'], predictions_series)
        mape = np.mean(np.abs((test['Bookings'] - predictions_series) / test['Bookings'])) * 100
        r2 = r2_score(test['Bookings'], predictions_series)
        
        # Store metrics
        results.append({
            'end_train': end_train,
            'p': p_fixed,
            'd': d_fixed,
            'q': q_fixed,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })
        
        # Store data for plotting
        end_train_values.append(end_train)
        mae_values.append(mae)
        mape_values.append(mape)
        r2_values.append(r2)
    except Exception as e:
        print(f"Failed to fit ARIMA({p_fixed},{d_fixed},{q_fixed}): {e}")

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)
# Print the results
print(results_df)

# Plotting section
plt.figure(figsize=(12, 16))

# Plot MAE, MAPE, and R2
plt.subplot(4, 1, 1)
plt.plot(end_train_values, mae_values, marker='o', linestyle='-', color='blue', label='MAE')
plt.title('Mean Absolute Error (MAE) Over Expanding Window Milano')
plt.xlabel('End of Training Set Index')
plt.ylabel('MAE')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(end_train_values, mape_values, marker='o', linestyle='-', color='orange', label='MAPE')
plt.title('Mean Absolute Percentage Error (MAPE) Over Expanding Window Milano')
plt.xlabel('End of Training Set Index')
plt.ylabel('MAPE')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(end_train_values, r2_values, marker='o', linestyle='-', color='green', label='R2')
plt.title('R-squared (R2) Over Expanding Window Milano')
plt.xlabel('End of Training Set Index')
plt.ylabel('R2')
plt.legend()

# Plot Actual and Predicted Values
plt.subplot(4, 1, 4)
for i in range(len(all_actual_values)):
    plt.plot(df.index, all_actual_values[i], label=f"Actual (Window {i + 1})", alpha=0.5)
    plt.plot(df.index, all_predicted_values[i], label=f"Predicted (Window {i + 1})", linestyle='dashed', alpha=0.7)

plt.title('Actual vs Predicted Values mle fitmodel Milano')
plt.xlabel('Hour')
plt.ylabel('Bookings')
plt.legend()

plt.tight_layout()
plt.show()


# In[19]:


####error metrics over total period
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Assuming hourly_counts is a dictionary with hours as keys and booking counts as values
df = pd.DataFrame(list(hourly_counts.items()), columns=['Hour', 'Bookings'])
df.set_index('Hour', inplace=True)

# Define the start of the test set
test_start = 504

# Initialize lists to store the true and predicted values for the whole period
true_values = []
predicted_values = []

# Initial training set size
initial_train_size = 168  # Start with the first week for training

# Fix values for ARIMA model
p_fixed = 4
d_fixed = 0
q_fixed = 3

# Expanding the window one day at a time
for end_train in range(initial_train_size, test_start, 12):  # Assuming each step adds one day (24 hours)
    train = df.iloc[:end_train]
    test = df.iloc[end_train:end_train+24]  # Next day for testing
    
    try:
        # Fit the ARIMA model with fixed values
        model = ARIMA(train, order=(p_fixed, d_fixed, q_fixed))
        fitted_model = model.fit()
        
        # Make predictions for the next day
        predictions = fitted_model.predict(start=train.index[-1] + 1, end=train.index[-1] + len(test))
        
        # Append the true values and predictions to the lists
        true_values.extend(test['Bookings'].values)
        predicted_values.extend(predictions)
        
    except Exception as e:
        print(f"Failed to fit ARIMA({p_fixed},{d_fixed},{q_fixed}): {e}")

# Now calculate the metrics over the entire period
mae = mean_absolute_error(true_values, predicted_values)
mape = np.mean(np.abs((np.array(true_values) - np.array(predicted_values)) / np.array(true_values))) * 100
r2 = r2_score(true_values, predicted_values)

# Print the aggregated metrics
print(f"MAE over entire period with expanding window Milano: {mae}")
print(f"MAPE over entire period with expanding window Milano: {mape}")
print(f"R^2 over entire period with expanding window Milano: {r2}")


# Import the necessary library for plotting
import matplotlib.pyplot as plt

# Convert true_values and predicted_values to a pandas Series for easy plotting, assuming index alignment
true_series = pd.Series(true_values, index=pd.RangeIndex(start=initial_train_size, stop=initial_train_size+len(true_values), step=1))
predicted_series = pd.Series(predicted_values, index=pd.RangeIndex(start=initial_train_size, stop=initial_train_size+len(predicted_values), step=1))

# Create a plot
plt.figure(figsize=(15, 7))  # Adjust the size as needed
plt.plot(true_series, label='Actual Bookings', color='blue', marker='o')  # Actual values
plt.plot(predicted_series, label='Predicted Bookings', color='red', linestyle='--')  # Predicted values
plt.title('Actual vs Predicted Bookings')
plt.xlabel('Hour')
plt.ylabel('Number of Bookings')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
plt.tight_layout()  # Adjust subplot parameters to give specified padding

# Show the plot
plt.show()


# In[ ]:




