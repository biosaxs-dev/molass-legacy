"""
    ABC.HistoryUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from .ParameterArray import ParameterArray

def get_min_dist_parameter(history, return_values=False):
    df, w = history.get_distribution(m=0, t=history.max_t)
    population = history.get_population(history.max_t)
    distance_df = population.get_weighted_distances()
    k = np.argmin(distance_df['distance'].values)
    values = df.iloc[k].values
    if return_values:
        return values
    return ParameterArray(values)