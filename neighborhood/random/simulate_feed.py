import os
import json
import requests
import pandas as pd
import datetime as dt
import collections

#this function accesses the weather api and writes the results to a local file
def get_weather(weather_file):
    weather_data = requests.get('https://api.darksky.net/forecast/a7c717ffebe004d187a6040f4c7f9c8d/40.714272,-74.005966')
    weather_json = weather_data.json()
    weather_json['read_time'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(weather_file, 'w') as outfile:
        json.dump(weather_json, outfile)

#primary function for this script
#this is to simulate live data
def simulate_feed(PULocationID):
    # set some parameters for target data
    days_ago = 365
    month_then = dt.datetime.now() - dt.timedelta(days=days_ago)
    month_then = month_then.strftime("%Y-%m")
    hour_then = dt.datetime.now().strftime(month_then+"-%d %H:00:00")

    #set read/write variables
    read_dir = 'neighborhood'+str(PULocationID)
    read_file = os.path.join('../', read_dir, 'ds-data-{}.csv'.format(month_then))
    weather_file = os.path.join('./','weather_data.json')

    if os.path.isfile(weather_file):
        #check when the last time weather data was refreshed, and refresh if necessary. 
        #This keeps us from exceeding our free api access limit of 1000 reads/day
        write_time = dt.datetime.fromtimestamp(os.path.getmtime(weather_file))
        time_now = dt.datetime.now()
        time_since_written = time_now - write_time

        #refresh data if it's an hour old
        if time_since_written > dt.timedelta(minutes=60):
            get_weather(weather_file)
    else:
        get_weather(weather_file)

    with open(weather_file, 'r') as json_file:
        weather_data = json.load(json_file)

    #grab max and min weather data
    max_temp = weather_data['daily']['data'][0]['temperatureMax']
    min_temp = weather_data['daily']['data'][0]['temperatureMin']

    #this is an estimate of total rainfall, and may not be very accurate. A better soluion may be to find the expected value
    #by using precipProbablity over the entire day. This'll do for now though.
    if 'precipIntensity' in weather_data['daily']['data'][0]: 
        precipitation = 24*weather_data['daily']['data'][0]['precipIntensity']
    else: 
        precipitation = 0
    if 'precipAccumulation' in weather_data['daily']['data'][0]: 
        snow = 24*weather_data['daily']['data'][0]['precipAccumulation']
    else:
        snow = 0

    datecol = 'tpep_pickup_datetime'
    neighborhood_data = pd.read_csv(read_file, index_col=[datecol], parse_dates=[datecol])
    number_of_pickups = neighborhood_data.loc[hour_then, "PULocationID"]

    return_list = collections.OrderedDict([
#        ('tpep_pickup_datetime', [hour_then]),
        ('number_of_pickups', [number_of_pickups]),
        ('PRCP', [precipitation]), ('SNOW', [snow]),
        ('TMAX', [max_temp]), ('TMIN', [min_temp])])

    return(return_list)
