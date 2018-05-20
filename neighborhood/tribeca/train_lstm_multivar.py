import os
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM

import sys
sys.path.insert(0, '../random/')
from utils import natsorted


def series2supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    # create input sequence (t - n, ..., t - 1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(-i))
        names += [('var{0}(t-{1})'.format(j+1, i)) for j in range(n_vars)]

    # create forcast sequence (t, t + 1, ..., t + n)
    for i in range(n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var{}(t)'.format(j+1)) for j in range(n_vars)]
        else:
            names += [('var{0}(t+{1})'.format(j+1, i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)

    return agg


def get_model(input_shape, model_loc=None):
    if model_loc is None:
        model = Sequential()
        model.add(LSTM(50, input_shape=input_shape))
        model.add(Dense(1))
    else:
        read_w = os.path.join(model_loc, 'weights.h5')
        read_arc = os.path.join(model_loc, 'arc.json')

        with open(read_arc, 'r') as a:
            model = model_from_json(a.read())
            model.load_weights(read_w)

    model.compile(loss='mae', optimizer='adam')

    return model


prd = '2016-05'
n_train_hours = int(744 * .75)  # 744 hours in may


neighborhoods = natsorted([dir for dir in os.listdir('../') if 'nei' in dir])
for nbh in neighborhoods:
    data_dir = os.path.join('../', nbh)
    model_loc = os.path.join(data_dir, 'model-multi')
    read = os.path.join(data_dir, 'ds-data-{}-multi.csv'.format(prd))

    dataset = pd.read_csv(read, header=0, usecols=[1, 2, 3, 4, 5])
    values = dataset.values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_values = scaler.fit_transform(values)
    # save scaler for predictions
    with open(os.path.join(model_loc, 'scaler.pkl'), 'wb') as pkl:
        pickle.dump(scaler, pkl, pickle.HIGHEST_PROTOCOL)

    reframed_values = series2supervised(scaled_values, 1, 1)

    # drop these columns so that we have reframed_values = [X, y]
    idx = [6, 7, 8, 9]
    reframed_values.drop(reframed_values.columns[idx], axis=1, inplace=True)

    train = reframed_values.values[:n_train_hours, :]
    test = reframed_values.values[n_train_hours:, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape into arrays of shape (sample, timesteps, features)
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    print('train_X.shape, train_y.shape, test_X.shape, test_y.shape')
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    model = get_model(train_X.shape[1:], model_loc=model_loc)

    history = model.fit(
        train_X, train_y, epochs=80, batch_size=72,
        validation_data=(test_X, test_y),
        verbose=2, shuffle=False)

    ###########################################################################
    yhat = model.predict(test_X)

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # invert forcast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)

    inv_yhat = inv_yhat[:, 0]

    # invert labels
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    eps = 5e-9
    mape = 100 * np.mean(np.abs((inv_y - inv_yhat)) / (inv_y + eps))
    print('Test Mape: {}%'.format(mape))

    # plt.figure()
    # plt.plot(inv_y[:500])
    # plt.plot(inv_yhat[:500])
    # plt.show()

    print('finished training: {}'.format(nbh))

    model.save_weights(os.path.join(model_loc, 'weights.h5'))

    with open(os.path.join(model_loc, 'arc.json'), 'w') as m:
        m.write(model.to_json())
