from pymongo import MongoClient
from keras.models import model_from_json
from utils import natsorted
import numpy as np
import pandas as pd
import os
import io

prd = '2017-04'

neighborhoods = natsorted([dir for dir in os.listdir('../') if 'hood' in dir])
for ii, nbh in enumerate(neighborhoods):
    model_location = os.path.join('..', nbh, 'model')

    arc = os.path.join(model_location, 'model-architecture.json')
    with open(arc, 'r') as m:
        model = model_from_json(m.read())
    
    model.load_weights(os.path.join(model_location, 'model-weights.h5'))

    s = io.StringIO()
    with open(os.path.join('..', nbh, 'ds-data-{}.csv'.format(prd)), 'r') as d:
        try:
            t = os.path.join(model_location, 'hr-of-month-{}.txt'.format(prd))
            itr = np.loadtxt(t)
        except:
            itr = 0

        next(d)
        s.write(next(line for num, line in enumerate(d) if num == itr))
