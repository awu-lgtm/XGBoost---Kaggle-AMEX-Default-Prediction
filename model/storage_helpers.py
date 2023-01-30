# functions that help reduce mem usage
import numpy as np


def partition(df, num_partitions=10):
    """
    Splits df into num_partitions
    Preserves groups of customer data
    """
    length = len(df.index)
    partitions = [i * length // num_partitions for i in range(1, num_partitions)]

    for i in range(len(partitions)):
        # add 1 to current partition if it is the same customer
        while (
            df.iloc[[partitions[i]]]["customer_ID"].values
            == df.iloc[[partitions[i] - 1]]["customer_ID"].values
        ):
            partitions[i] += 1

    partitions.append(len(df.index))
    print(partitions)
    return partitions


def df_slice(df, num_partitions):
    slices = []
    start_index = 0
    for p in partition(df, num_partitions):
        slices.append(df.iloc[start_index:p])
        start_index = p
    return slices


def reduce_mem_usage(df):
    """Reduces memory usage by converting float64 columns to float32"""
    i = 1
    cols = []
    for col in df.columns:
        if i % 1000 == 0:
            print(i)
        i += 1

        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif col_type == "float64":
                df[col] = df[col].astype(np.float32)
    return cols
