import numpy as np
import pandas as pd

def result_writer(indices_list: list, result_list: list, meta: pd.DataFrame, column):
    flat_indices = []
    for x in zip(indices_list):
        flat_indices.extend(x)
    flat_results = []
    for x in zip(result_list):
        flat_results.extend(x)
    
    flat_indices = np.array(flat_indices)
    flat_results = np.array(flat_results)

    unique_indices, unique_indices_idx = np.unique(flat_indices, return_index=True)
    meta.loc[unique_indices, column[0]] = flat_results[unique_indices_idx]

    meta = meta.loc[unique_indices]
    return meta