from simulate_feed import simulate_feed
from pymongo import MongoClient
from sklearn.cluster import KMeans
from keras.models import model_from_json
from utils import natsorted
import numpy as np
import pandas as pd
import datetime
import pickle
import os


prd = '2017-05-multi'

agg = []
neighborhoods = natsorted([dir for dir in os.listdir('../') if 'hood' in dir])
for id, nbh in enumerate(neighborhoods, 1):
    model_location = os.path.join('..', nbh, 'model-multi')

    with open(os.path.join(model_location, 'arc.json'), 'r') as m:
        model = model_from_json(m.read())
    
    with open(os.path.join(model_location, 'scaler.pkl'), 'rb') as pkl:
        scaler = pickle.load(pkl)

    model.load_weights(os.path.join(model_location, 'weights.h5'))

    print('about to fetch data for {}'.format(nbh))
    crt = pd.DataFrame(simulate_feed(id)).values  # shape (1, 5)

    model.compile(loss='mae', optimizer='adam')

    scaled_crt = scaler.transform(crt)
    scaled_p = model.predict(scaled_crt[np.newaxis, :])

    prediction = scaler.inverse_transform(np.broadcast_to(scaled_p, (1, 5)))
    prediction = prediction[:, 0]

    print('prediction =', prediction)

    agg.append(pd.DataFrame(prediction))


timeframe = pd.concat(agg, axis=0, ignore_index=True)

timeframe.index += 1
timeframe.index.rename('n_id', inplace=True)
timeframe.columns = ['count']

timeframe['color'] = np.nan

kmeans = KMeans(n_clusters=5).fit(timeframe['count'].values[:, np.newaxis])
idx = kmeans.labels_
sorted_idx = np.argsort(kmeans.cluster_centers_, axis=None)

colors = ['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f', '#006d2c']
for ii, color in zip(sorted_idx, colors):
    timeframe.loc[idx == ii, 'color'] = color

pw = 'lost$tarling'
url = 'ec2-34-227-52-163.compute-1.amazonaws.com'
client = MongoClient('mongodb://mikebmc:{0}@{1}:27017/'.format(pw, url))

db = client.meteor

timeframe = timeframe.reset_index().to_dict(orient='list')
post_id = db.toplayer.update({}, timeframe, upsert=True)

print('uploaded new timeframe {}'.format(datetime.datetime.now()))
