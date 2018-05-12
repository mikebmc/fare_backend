import pandas as pd
import os
from utils import cleanup, natsorted


prd = '2016-05-multi'
nbh_dir = '../neighborhood'
data_dir = '../../data/'
read = os.path.join(data_dir, 'fixed_pickups_cut_yellow_tripdata_2016-05.csv')


# delete files if they already exist
neighborhoods = natsorted([dir for dir in os.listdir('../') if 'nei' in dir])
for nbh in neighborhoods:
    write = os.path.join('../', nbh, 'data-{}.csv'.format(prd))
    cleanup(write)


to_drop = ['month', 'weekday', 'Sunday', 'Monday', 'Tuesday',
           'Wednesday', 'Thursday', 'Friday', 'Saturday']


for count, chunk in enumerate(pd.read_csv(read, chunksize=10 ** 6), 1):
    chunk.drop(to_drop, axis=1, inplace=True)

    groups = chunk.groupby('LocationID')
    for name, grp in groups:
        write = os.path.join(nbh_dir + str(name), 'data-{}.csv'.format(prd))

        if os.path.exists(write):
            grp.to_csv(write, mode='a', header=False, index=False)
        else:
            grp.to_csv(write, mode='w', index=False)

    print("Processed chunk: {}".format(count))
