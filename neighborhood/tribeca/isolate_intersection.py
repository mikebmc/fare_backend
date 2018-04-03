import pandas as pd
import numpy as np
from pathlib import Path
import os

read_file = 'tribeca_data_2016-06.csv'
write_file = Path('busy_intersection.csv')

if write_file.is_file():
    os.remove(write_file)

intersection = np.array([40.720768, -74.010015])
half_block_from_int = np.array([40.720410, -74.010097])

radius = np.linalg.norm(half_block_from_int - intersection)
print('radius =', radius)

chunksize = 10 ** 6
for pickups in pd.read_csv(read_file, chunksize=chunksize):
    translated_pickups = pickups.loc[
        :, ['pickup_latitude', 'pickup_longitude']] - intersection
    distance_to_intersection = np.linalg.norm(translated_pickups, axis=1)
    pickups = pickups.loc[distance_to_intersection < radius]

    if write_file.is_file():
        pickups.to_csv(write_file, mode='a', header=False, index=False)
    else:
        pickups.to_csv(write_file, mode='w', index=False)

print('Saved file {}'.format(write_file))
