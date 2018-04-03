import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('busy_intersection.csv')

# make index = timestamp
data.index = pd.to_datetime(data['tpep_pickup_datetime'])
data.drop('tpep_pickup_datetime', axis=1, inplace=True)

# drop lat lon
data.drop(['pickup_longitude', 'pickup_latitude'], axis=1, inplace=True)

# add count column
data['count'] = 1

# resample on 15min basis, summing occurences
data = data.resample('30T').sum()
print('data =\n{}'.format(data))

plt.plot(data)
plt.xlabel('30 minute intervals')
plt.ylabel('number of pickups')
plt.show()
