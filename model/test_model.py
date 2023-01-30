import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from constants import *
from storage_helpers import *
from pre_processing import process_data


def process_test_data(df, num_partitions=10):
    test_preds = []

    start_index = 0
    count = 1

    for p in partition(df, num_partitions):
        X = process_data(df.iloc[start_index:p])[FEATURES]
        reduce_mem_usage(X)
        X = X.reset_index()

        dtest = xgb.DMatrix(data=X[X.columns[1::]])
        del X
        gc.collect()

        model = xgb.Booster()
        model.load_model(f"XGB_fold0.xgb")
        preds = model.predict(dtest)
        for fold in range(1, FOLDS):
            model.load_model(f"XGB_fold{fold}.xgb")
            preds += model.predict(dtest)
        preds /= FOLDS
        test_preds.append(preds)

        start_index = p
        print(count)
        count += 1

        del dtest, model
        gc.collect()

    return test_preds


test = pd.read_parquet(TEST_PATH, engine="pyarrow")
test["S_2"] = pd.to_datetime(test["S_2"])

test_preds = process_test_data(test)
unique_ids = test["customer_ID"].unique()

del test
gc.collect()

test_preds = np.concatenate(test_preds)
submission = pd.DataFrame(index=unique_ids, data={"prediction": test_preds})
submission.index.names = ["customer_ID"]
submission.reset_index(inplace=True)
submission.to_csv(TEST_OUTPUT_PATH, index=False)
submission.head()
