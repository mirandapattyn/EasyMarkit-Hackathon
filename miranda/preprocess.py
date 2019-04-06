import numpy as np
import pandas as pd


def check_unique_values(data):
    for col in data.columns.values:
        print(col + ': ' + str(data[col].nunique()))


def one_hot_encode(raw_data, upper_limit=20, lower_limit=0, debug=False):
    new_data = pd.DataFrame({})
    for data in raw_data:
        col_type = raw_data[data].dtype
        col = raw_data[data]
        if col_type == 'object' or lower_limit < col.nunique() < upper_limit:
            one_hot = pd.get_dummies(col, prefix=data)
            new_data = pd.concat([new_data, one_hot], axis=1)
        else:
            new_data = pd.concat([new_data, col], axis=1)
    if debug:
        new_data.info()
        print(new_data.shape)
    return new_data


if __name__ == '__main__':
    df = pd.read_csv("data/train.csv")
    check_unique_values(df)
    df_onehot = one_hot_encode(df, debug=True)

    print(df_onehot)
