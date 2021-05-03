import pandas as pd
import numpy as np
import random

def one_hot_encode_column(df, column, prefix):
    return pd.concat([pd.get_dummies(df[column], prefix=prefix), df], axis=1)

def norm_column(df, column):
    new_df = df.copy()

    max_value = new_df[column].sort_values().iloc[-1]

    return new_df[column] / max_value

def fill_missing_values(df, column):
    
    values = df[column].dropna().to_list()

    def fill(value):
        if type(value) == float and np.isnan(value):
            return random.choice(values)
        else:
            return value

    new_df = df.copy()
    new_df[column] = new_df[column].apply(fill)

    return new_df