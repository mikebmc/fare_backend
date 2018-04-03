import pandas as pd
import os
from pathlib import Path

# define geofence around tribeca. Currently this is a square, King has a
# linear solution to make it more precise in the works
tribeca = pd.read_csv('tribeca_boundries.csv')
lat_max = tribeca.tribeca_lat.max()
lat_min = tribeca.tribeca_lat.min()
lon_max = tribeca.tribeca_lon.max()
lon_min = tribeca.tribeca_lon.min()

# set files that we will read from and written to
write_file = Path('tribeca_data_2016-06.csv')
read_file = 'yellow_tripdata_2016-06.csv'

# if write_file exists, remove it to prevent duplicate data
if write_file.is_file():
    os.remove(write_file)


def process(chunk, verbose=False):
    # remove the columns that we don't care about
    chunk = chunk.drop(['VendorID', 'tpep_dropoff_datetime', 'passenger_count',
                        'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
                        'dropoff_longitude', 'dropoff_latitude',
                        'payment_type', 'fare_amount', 'extra', 'mta_tax',
                        'tip_amount', 'tolls_amount', 'improvement_surcharge',
                        'total_amount'], axis=1)

    # find and remove any entries outside of geofence area around tribeca
    in_lat = (chunk['pickup_latitude'] < lat_max) & (
        chunk['pickup_latitude'] > lat_min)
    in_lon = (chunk['pickup_longitude'] < lon_max) & (
        chunk['pickup_longitude'] > lon_min)
    chunk = chunk[in_lat & in_lon]

    # write trips to a new csv file for use in next script, we append if the
    # file exists since we are in a loop and dealing with chunks
    if write_file.is_file():
        chunk.to_csv(write_file, mode='a', header=False, index=False)
    else:
        chunk.to_csv(write_file, mode='w', index=False)

    if verbose:
        print(chunk.head())


# break up the input file into more manageable chunks
chunksize = 10 ** 6
for chunk in pd.read_csv(read_file, chunksize=chunksize):
    process(chunk)
