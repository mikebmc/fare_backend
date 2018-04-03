import pandas as pd
import numpy as np
from scipy.linalg import circulant
import os

# tribeca = pd.read_csv('tribeca_boundries.csv').values.astype('float32').T

# edges = tribeca.dot(circulant([-1, 1, 0, 0]))
# orths = np.array([[0, 1], [-1, 0]]).dot(edges)
# struct = (edges, orths)

# data_dir = '../../data/'
# write_data = os.path.join(data_dir, 'tribeca_data_2016-06.csv')
# read_data = os.path.join(data_dir, 'yellow_tripdata_2016-06.csv')

# try:
#     os.remove(write_data)
# except OSError:
#     pass


def _get_edges_orths(vertices):
    edges = vertices.dot(circulant([-1, 1, 0, 0]))
    orthogonals = np.array([[0, 1], [-1, 0]]).dot(edges)
    return (edges, orthogonals)


class filter:
    def __init__(self, corners_counterclockwise, filename, verbose=False):
        self.corners = corners_counterclockwise
        self.filename = filename
        self.file_init = False
        self.struct = _get_edges_orths(self.corners)
        self.verbose = verbose

    def open(self):
        # delete filename if exists, else do nothing
        try:
            os.remove(self.filename)
        except OSError:
            pass

    def get_write_inner(self, chunk):
        # Filter away points not inside a convex polygon
        # Inputs:
        #  - chunk   -- dataframe containing lat lon
        #  - tup     -- tuple containing polygon-edges and inward-orth-vectors
        #  - verbose -- print chunk.head() if true
        edg, ort = self.struct
        latlon = chunk[['pickup_latitude', 'pickup_longitude']
                       ].values.astype('float32')

        index = np.all(latlon.dot(ort) -
                       np.sum(ort * self.corners, axis=0) > 0, axis=1)
        chunk = chunk.loc[index]

        if self.verbose:
            print("Processed chunk:")
            print(chunk.head())

            if os.path.exists(self.filename):
                chunk.to_csv(self.filename, mode='a',
                             header=False, index=False)
            else:
                chunk.to_csv(self.filename, mode='w', index=False)

    def run(self, chunk):
        self.get_write_inner(chunk)
