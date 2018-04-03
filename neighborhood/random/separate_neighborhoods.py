import pandas as pd
import os
from utils import cleanup

neighborhood_dir = '../neighborhood'
data_dir = '../../data/'
prd = '2017-03'
read = os.path.join(data_dir, 'yellow_tripdata_{}.csv'.format(prd))

# delete all matching files
for nh in range(1, 266):
    write = os.path.join(neighborhood_dir + str(nh), 'data-{}.csv'.format(prd))
    cleanup(write)

to_drop = ['VendorID', 'tpep_dropoff_datetime', 'passenger_count',
           'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
           'payment_type', 'fare_amount', 'extra', 'mta_tax',
           'tip_amount', 'tolls_amount', 'improvement_surcharge',
           'total_amount', 'DOLocationID']

count = 1
chunksize = 10 ** 6
for chunk in pd.read_csv(read, chunksize=chunksize):
    chunk.drop(to_drop, axis=1, inplace=True)
    groupS = chunk.groupby('PULocationID')
    for name, group in groupS:
        write = os.path.join(neighborhood_dir + str(name),
                             'data-{}.csv'.format(prd))
        if os.path.exists(write):
            group.to_csv(write, mode='a', header=False, index=False)
        else:
            group.to_csv(write, mode='w', index=False)

    print("Processed chunk {}:".format(count))
    count += 1
