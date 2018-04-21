import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for ii in range(len(dataset) - look_back - 1):
        a = dataset[ii:ii + look_back, :]
        dataX.append(a)
        dataY.append(dataset[ii + look_back, 0])
    return np.array(dataX), np.array(dataY)


def do_stuff(input_data, look_back=1):
    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_data = scaler.fit_transform(input_data)

    # split into train and test
    train, test, y_train, y_test = train_test_split(
        input_data, input_data, test_size=.67, random_seed=117
    )

    # train_size = int(len(input_data) * 0.67)
    # train, test = input_data[:train_size, :], input_data[train_size:, :]

    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    hist = model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=2)

    # make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])

    # calculate root mean squared error
    print('Train Score: {} RMSE'.format(hist.history['mean_squared_error']))
    test_score = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
    print('Test Score: {} RMSE'.format(test_score))

    plot_first_week(input_data, look_back, train_predict, test_predict)

    return model


def plot_first_week(input_data, look_back, train_predict, test_predict):
    # shift train predictions for plotting
    train_predict_plot = np.empty_like(input_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(
        train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    test_predict_plot = np.empty_like(input_data)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) +
                      1:len(input_data) - 1, :] = test_predict

    # plot baseline and predictions
    days_of_week = ('Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat')
    plt.xlim(24, 192)
    plt.xticks(np.arange(36, 204, step=24), days_of_week)
    plt.plot(data)
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show()


# data = pd.read_csv('_30_minute_intersection.csv', usecols=[1],
#                    sep=',').values.astype('float32').reshape(-1)

numzones = 265
for neighborhood in range(257, numzones + 1):
    data_dir = '../neighborhood' + str(neighborhood)
    print(data_dir)

    read = os.path.join(data_dir, 'ds-data-2017-05.csv')
    data = pd.read_csv(read, usecols=[1], sep=',').values.astype(
        'float32').reshape(-1, 1)
    
    model = do_stuff(data, look_back=1)

    save_location = os.path.join(data_dir, 'model')
    model.save_weights(os.path.join(save_location, 'model-weights.h5'))
    with open(
            os.path.join(save_location, 'model-architecture.json'), 'w'
    ) as save:
        save.write(model.to_json())
