import pandas as pd
import os


def delete_old_file(f):
    try:
        os.remove(f)
    except OSError:
        pass


def write_new_file(f):
    pass


def swap_columns(r, w, d, col):
    read = os.path.join(d, r)
    write = os.path.join(d, w)

    delete_old_file(write)
    
    for i, chunk in enumerate(pd.read_csv(read, chunksize=10 ** 6)):
        chunk = chunk.reindex(columns=col)
        if os.path.exists(write):
            chunk.to_csv(write, mode='a', header=None, index=False)
        else:
            chunk.to_csv(write, mode='w', index=False)

        print('processed chunk: {}'.format(i))


read = 'pickups_cut_yellow_tripdata_2016-05.csv'
write = 'fixed_pickups_cut_yellow_tripdata_2016-05.csv'
data_dir = '../../data'

columns = ['tpep_pickup_datetime', 'LocationID', 'month', 'weekday',
           'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
           'Saturday', 'PRCP', 'SNOW', 'TMAX', 'TMIN']

swap_columns(read, write, data_dir, columns)
