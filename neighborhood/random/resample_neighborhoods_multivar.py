import os
import pandas as pd
import datetime
import calendar
from utils import natsorted


prd = '2016-05-multi'
datecol = 'tpep_pickup_datetime'


# get the first day of the month and use that to build a range object
# datetime.datetime(year, month, day)
first = datetime.datetime(*[int(x) for x in prd.split('-')[:-1]], 1)
hrs_in_month = 24 * calendar.monthrange(first.year, first.month)[1]
idx = pd.date_range(first, name=datecol, periods=hrs_in_month, freq='H')


noa_file = 'NOAA_data_2016-05.csv'
noa = pd.read_csv(os.path.join('../../data/', noa_file),
                  index_col=['DATE'], parse_dates=['DATE'])
noa.index.rename('tpep_pickup_datetime', inplace=True)


neighborhoods = natsorted([dir for dir in os.listdir('../') if 'nei' in dir])
for nbh in neighborhoods:
    read = os.path.join('../', nbh, 'data-{}.csv'.format(prd))
    write = os.path.join('../', nbh, 'ds-data-{}.csv'.format(prd))

    try:
        data = pd.read_csv(read, index_col=[datecol], parse_dates=[datecol])
        data = data.resample('H').count().reindex(idx, fill_value=0)
    except:
        data = pd.DataFrame(columns=['LocationID'], index=idx).fillna(0)
        print("{} had no pickup data and was zero filled".format(nbh))

    data = pd.concat([data, noa], axis=1)
    data.fillna(method='ffill').to_csv(write, index=True)
