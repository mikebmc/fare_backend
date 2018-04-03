import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

all_pickups_data = pd.read_csv('yellow_tripdata_2016-01.csv')
sub_pickups_data = all_pickups_data.iloc[:,:-9].values

sub_pickups_data.head()
