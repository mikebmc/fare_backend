import os
import pandas as pd
import datetime
import calendar
from utils import natsorted


prd = '2016-05-multi'
datecol = 'tpep_pickup_datetime'


# get the first day of the month and use that to build a range object
# datetime.datetime(year, month, day)
first = datetime.datetime(*[int(x) for x in prd.split('-')][:-1], 1)
hrs_in_month = 24 * calendar.monthrange(first.year, first.month)[1]
idx = pd.date_range(first, name=datecol, periods=hrs_in_month, freq='H')


neighborhoods = natsorted([dir for dir in os.listdir('../') if 'nei' in dir])
for nbh in neighborhoods:
    read = os.path.join('../', nbh, 'data-{}.csv'.format(prd))
    write = os.path.join('../', nbh, 'ds-data-{}'.format(prd))

    try:
        data = pd.read_csv(read, index_col=[datecol], parse_dates=[datecol])
        data = data.resample('H').count().reindex(
            idx, fill_value=0).to_csv(write, index=True)
    except:
        pd.DataFrame(columns=['LocationID'],
                     index=idx).fillna(0).to_csv(write)
        print("{} had no pickup data and has been filled with 0's".format(nbh))
