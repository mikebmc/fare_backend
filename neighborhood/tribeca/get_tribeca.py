from filter_manager import filter
import pandas as pd
import os

latlon = pd.read_csv('tribeca_boundries.csv').values.astype('float32').T

data_dir = '../../data/'
write = os.path.join(data_dir, 'tribeca_data_2016-06.csv')
read = os.path.join(data_dir, 'yellow_tripdata_2016-06.csv')

tribeca = filter(latlon, write, verbose=True)
tribeca.open()

to_drop = ['VendorID', 'tpep_dropoff_datetime', 'passenger_count',
           'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
           'dropoff_longitude', 'dropoff_latitude',
           'payment_type', 'fare_amount', 'extra', 'mta_tax',
           'tip_amount', 'tolls_amount', 'improvement_surcharge',
           'total_amount']

chunksize = 10 ** 6
for chunk in pd.read_csv(read, chunksize=chunksize):
    chunk.drop(to_drop, axis=1, inplace=True)
    tribeca.run(chunk)
