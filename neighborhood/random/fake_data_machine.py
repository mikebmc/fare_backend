import pandas as pd
import numpy as np

fake_timeframe = pd.DataFrame({
    'n_id': range(1, 266),
    'count': np.random.randint(
        low=0, high=45, size=(265,))
})

columns = fake_timeframe.columns[::-1]
fake_timeframe = fake_timeframe.reindex(columns=columns)

fake_timeframe.to_csv('fake_timeframe.csv', index=False)
