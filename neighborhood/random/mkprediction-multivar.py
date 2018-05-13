from pymongo import MongoClient
from keras.models import model_from_json
from utils import natsorted
import numpy as np
import pandas as pd
import os
import io

prd = '2017-05-multi'

agg =[]
neighborhoods = natsorted([dir for dir in os.listdir('../') if 'hood' in dir])
for ii, nbh in enumerate(neighborhoods, 1):
    model_location = os.path.join('..', nbh, 'model-multi')

    with open(os.path.join(model_location, 'arc.json'), 'r') as m:
        model = model_from_json(m.read())
    
    model.load_weights(os.path.join(model_location, 'weights.h5'))

    # {'tpep_pickup_datetime': '2017-05-12 17:00:00', 'number_of_pickups': 376, 'PRCP': 0.4416, 'SNOW': 0, 'TMAX': 66.07, 'TMIN': 53.2}
