import numpy as np
import pandas as pd

def create_features(y_window, last_date, step_h):
    features = list(y_window)
    target_date = last_date + pd.Timedelta(hours=step_h)

    features.extend([
        target_date.hour,
        target_date.dayofweek,
        target_date.day
    ])
    return np.array(features)
