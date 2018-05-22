import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

target = '../neighborhood90/ds-data-2017-04.csv'

df = pd.read_csv(target, index_col=0)
print(df.head(5))

plt.plot(df['PULocationID'].astype(np.float32))
plt.show()
