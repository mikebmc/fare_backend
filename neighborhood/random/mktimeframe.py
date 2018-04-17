import io
import os
import re
import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils import natsorted
from pymongo import MongoClient

min_interval = 60
prd = '2017-04'

# get the date from 1 year ago ignoring leap years
dt = datetime.datetime.now() - datetime.timedelta(days=365)

# round down to the nearest minute divisible by min_interval
num_minutes = dt.minute // min_interval
dt = dt.replace(minute=num_minutes * min_interval,
                second=0).strftime('%Y-%m-%d %H:%M:%S')

# get a naturally sorted list of the neighborhood# directories
neighborhoods = natsorted([dir for dir in os.listdir('../') if 'neigh' in dir])

# aggregate the data from all our csv's
agg = []
for nbh in neighborhoods:
    read = os.path.join('../', nbh, 'ds-data-{}.csv'.format(prd))
    s = io.StringIO()
    with open(read, 'r') as ff:
        s.write(next(ff))
        s.write(next(line for line in ff if re.search('^{}'.format(dt), line)))
        s.seek(0)
    agg.append(pd.read_csv(s, usecols=['PULocationID']))

# combine all of the data matching today's datetime and drop date info
timeframe = pd.concat(agg, axis=0, ignore_index=True)

# the new index is a range beginning at 1
timeframe.index += 1
timeframe.index.name = 'n_id'
timeframe.rename(columns={'PULocationID': 'count'}, inplace=True)
timeframe['color'] = np.nan

# use kmeans to calculate centroids for color assignments and sort them
kmeans = KMeans(n_clusters=5).fit(timeframe['count'].values.reshape(-1, 1))
idx = kmeans.labels_
sorted_idx = np.argsort(kmeans.cluster_centers_, axis=None)

for ii, color in zip(
        sorted_idx,
        ['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f', '#006d2c']):
    timeframe.loc[idx == ii, 'color'] = color

# open a connection to mongodb
client = MongoClient('mongodb://localhost:3001/')
db = client.meteor

# and replace what's currently inside of the toplayer collection
post_id = db.toplayer.update(
    {}, timeframe.reset_index().to_dict(orient='list'),
    upsert=True)

# tell me the datetime of the data pushed
print('dt =\n{}'.format(dt))
