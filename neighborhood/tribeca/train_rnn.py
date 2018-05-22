import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import os


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
    plt.plot(input_data)
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show()


def plot_target_and_prediction(input_data, prediction, start=0):
    vector_sz = len(prediction)
    input_data = input_data[start:start + vector_sz]
    plt.plot(input_data)
    plt.plot(prediction)
    plt.show()


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for ii in range(len(dataset) - look_back - 1):
        a = dataset[ii:ii + look_back, :]
        dataX.append(a)
        dataY.append(dataset[ii + look_back, 0])
    return np.array(dataX), np.array(dataY)


def do_stuff(input_data, look_back=1, model=None):
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_data = scaler.fit_transform(input_data)

    # normalize data, and split into train and test
    train_size = int(len(input_data) * 0.67)
    train, test = input_data[:train_size, :], input_data[train_size:, :]

    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    if model is None:
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(X_train, y_train, epochs=15, batch_size=1, verbose=2)

    # make predictions
    train_predict = model.predict(X_train)
    # test_predict = model.predict(X_test)

    test_predict = y_test.reshape((-1, 1))
    # train_predict = y_train.reshape((-1, 1))

    # invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])
    input_data = scaler.inverse_transform(input_data)

    # calculate root mean squared error
    # train_score = np.sqrt(mean_squared_error(
    #     y_train[0], train_predict[:, 0]))

    eps = 5e-9
    train_score = 100 * np.mean(np.abs((
        y_train[0] - train_predict[:, 0]) / (y_train[0] + eps)))
    print('Train Score: {}% error'.format(train_score))
    # test_score = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
    test_score = 100 * np.mean(np.abs((
        y_test[0] - test_predict[:, 0]) / (y_test[0] + eps)))
    print('Test Score: {}% error'.format(test_score))

    # plot_first_week(input_data, look_back, train_predict, test_predict)
    prediction = [model.predict(X_train[ii:ii + 1]) for ii in range(48)]
    prediction = np.concatenate(prediction, axis=1).T
    prediction = scaler.inverse_transform(prediction)
    plot_target_and_prediction(input_data, prediction[:24], start=0)
    quit()
    return model


numzones = 265
for neighborhood in range(100, numzones + 1):
    data_dir = '../neighborhood' + str(neighborhood)
    print(data_dir)

    read = os.path.join(data_dir, 'ds-data-2017-05.csv')
    data = pd.read_csv(read, usecols=[1], sep=',').values.astype(
        'float32').reshape(-1, 1)

    save_location = os.path.join(data_dir, 'model')

    with open(os.path.join(
            save_location, 'model-architecture.json'), 'r') as m:
        model = model_from_json(m.read())

    model.load_weights(os.path.join(save_location, 'model-weights.h5'))
    model = do_stuff(data, look_back=1, model=model)

    # model.save_weights(os.path.join(save_location, 'model-weights.h5'))
    # with open(os.path.join(
    #         save_location, 'model-architecture.json'), 'w') as save:
    #     save.write(model.to_json())
