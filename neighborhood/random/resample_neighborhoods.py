import os
import pandas as pd
import datetime
import calendar
from utils import natsorted


prd = '2017-03'  # 'yyyy-mm'
datecol = 'tpep_pickup_datetime'


# get the first day of the month and use that to build a range object
first = datetime.datetime(*[int(x) for x in prd.split('-')], 1)
hrs_in_month = 24 * calendar.monthrange(first.year, first.month)[1]
idx = pd.date_range(first, name=datecol, periods=hrs_in_month, freq='H')


# get a naturally sorted list of neigborhood# directories
neighborhoods = natsorted([dir for dir in os.listdir('../') if 'neigh' in dir])


# downsample the read data found in each neigborhood# directory if it exists
for nbh in neighborhoods:
    read = os.path.join('../', nbh, 'data-{}.csv'.format(prd))
    write = os.path.join('../', nbh, 'ds-data-{}.csv'.format(prd))

    try:
        data = pd.read_csv(read, index_col=[datecol], parse_dates=[datecol])
        data = data.resample('H').count().reindex(
            idx, fill_value=0).to_csv(write, index=True)
    except:
        pd.DataFrame(columns=['PULocationID'],
                     index=idx).fillna(0).to_csv(write)
        print('{} has no data'.format(nbh))
