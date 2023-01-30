# pre-processing
import gc
import pandas as pd
from constants import *
from storage_helpers import *


def process_data(df):
    """data preprocessing"""
    g = df.groupby("customer_ID")
    n_c = g[numerical_and_categorical_columns]

    p = pd.concat(
        [basic_agg(n_c), f_l_m(n_c), lag(n_c), interval_stats(g), rolling_stats(g)],
        axis=1,
    )

    def basic_agg(n_c):
        """aggregation with standard dev, min, max, sum"""
        agg = n_c.agg(["std", "min", "max", "sum"])
        agg.columns = ["_".join(column) for column in agg.columns]
        return agg

    def f_l_m(n_c):
        """first, last, mean, lag - first, mean - last, lag1 - last"""
        f = n_c.first()
        l = n_c.last()
        m = n_c.mean()
        lag = n_c.nth(-2)

        l_f = l - f
        m_l = m - l
        lag_l = lag - l

        rename_columns(
            [f, l, m, l_f, m_l, lag_l],
            [
                "first",
                "last",
                "mean",
                "last_sub_first",
                "mean_sub_last",
                "lag_sub_last",
            ],
        )
        return pd.concat([f, l, m, l_f, m_l, lag_l], axis=1)

    def lag(n_c):
        """lag1 to lag5 (6 months before)"""
        l = n_c.nth(-2)
        l.columns = [f"{col}_lag_{1}" for col in l.columns]
        for i in range(2, 5):
            tmp = n_c.nth(-i - 1)
            tmp.columns = [f"{col}_lag_{i}" for col in tmp.columns]
            l = pd.concat([l, tmp], axis=1)
        return l

    def interval_stats(g):
        "mean and standard deviation in intervals"

        def standard_deviation(i, m, tail):
            x_m = tail.set_index("customer_ID", drop=True) - m
            summation = (x_m**2).groupby("customer_ID").sum(min_count=i)
            d = (summation / (i - 1)) ** 0.5
            return d

        def mean(t, i):
            m = t.sum(min_count=i) / i
            return m

        result = None
        for i in [2, 3, 6, 13, 15]:
            tail = g.tail(i)
            t = tail.groupby("customer_ID")
            u = t[numerical_and_categorical_columns]

            m = mean(u, i)
            d = standard_deviation(u, i, m, tail)

            rename_columns([m, d], ["mean", "std"], [str(i)] * 2)

            if result is None:
                result = pd.concat([m, d], axis=1)
            else:
                result = pd.concat([result, m, d], axis=1)
        return result

    def rolling_stats(g):
        """sum and delta"""

        def sum_interval(t, i):
            s = t.sum(min_count=len(i))
            return s

        def delta_interval(t, i):
            d = t.first() - t.last()
            return d

        def filter_count(df, t, i):
            """filters out any customers who has less than i data points"""
            c = t.count()
            return df[c.iloc[:, 1] >= len(i)]

        result = None
        for i in [
            [-1, -2],
            [-1, -2, -3],
            [-4, -5, -6],
            [-7, -8, -9, -10, -11, -12, -13],
        ]:
            t = g.nth(i).groupby("customer_ID")[numerical_and_categorical_columns]
            s = sum_interval(t, i)
            d = delta_interval(t, i)

            d = filter_count(d, t, i)

            i_string = "_".join([f"{-n}" for n in i])
            rename_columns([s, d], ["sum", "delta"], [i_string] * 2)

            if result is None:
                result = pd.concat([s, d], axis=1)
            else:
                result = pd.concat([result, s, d], axis=1)
        return result

    def rename_columns(dfs, names, extra=None):
        if extra is None:
            for df, name in zip(dfs, names):
                df.columns = [f"{col}_{name}" for col in df.columns]
        else:
            for df, name, e in zip(dfs, names, extra):
                df.columns = [f"{col}_{name}_{e}" for col in df.columns]

    return p


def process(slices):
    processed = None
    i = 0
    count = 1
    while i < len(slices):
        df_slice = process_data(slices[i])[FEATURES]
        if processed is None:
            processed = df_slice
        else:
            processed = pd.concat([processed, df_slice], axis=0)
        slices.pop(0)

        print(count)
        count += 1

        del df_slice
        gc.collect()
    return processed


y = pd.read_csv(TRAIN_LABELS_PATH)
X = pd.read_parquet(TRAIN_PATH)
X["S_2"] = pd.to_datetime(X["S_2"])

df_slices = df_slice(X, 10)  # used to prevent running out of ram

del X
gc.collect()

X = process(df_slices)
reduce_mem_usage(X)

y = y.set_index("customer_ID", drop=True)
train = X.merge(y, left_index=True, right_index=True, how="left")
train.reset_index(inplace=True)
train.head()

del X, y
gc.collect()

train.to_parquet(path=PROCESSED_OUTPUT_PATH)  # save training data for model creation
