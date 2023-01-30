import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from constants import *
import cudf
import cupy

xgb_params = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "eval_metric": "logloss",
    "objective": "binary:logistic",
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "random_state": 42,
}


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256 * 1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0
        self.batch_size = batch_size
        self.batches = int(np.ceil(len(df) / self.batch_size))
        super().__init__()

    def reset(self):
        """Reset the iterator"""
        self.it = 0

    def next(self, input_data):
        """Yield next batch of data."""
        if self.it == self.batches:
            # Return 0 when there's no more batch.
            return 0
        a = self.it * self.batch_size
        b = min((self.it + 1) * self.batch_size, len(self.df))
        df = cudf.DataFrame(self.df.iloc[a:b])
        input_data(data=df[self.features], label=df[self.target])
        self.it += 1
        return 1


train = pd.read_parquet(path=PROCESSED_OUTPUT_PATH)

weights = []
skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# 5 fold KFold CV
for fold, (train_idx, valid_idx) in enumerate(skf.split(train, train.target)):
    Xy_train = IterLoadForDMatrix(train.loc[train_idx], FEATURES, "target")

    X_valid = train.loc[valid_idx, FEATURES]
    y_valid = train.loc[valid_idx, "target"]

    dtrain = xgb.DeviceQuantileDMatrix(
        Xy_train, max_bin=256
    )  # use DeviceQuantileDMatrix to save memory
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid)

    model = xgb.train(
        xgb_params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        num_boost_round=9999,
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    model.save_model(f"XGB_fold{fold}.xgb")

    weight = model.get_score(
        importance_type="weight"
    )  # compute weights of each feature
    weights.append(
        pd.DataFrame({"feature": weight.keys(), f"importance_{fold}": weight.values()})
    )

    del dtrain, Xy_train, weight
    del X_valid, y_valid, dvalid, model
    gc.collect()

# Feature selection
df = None
for fw in weights:  # fw: feature_weights
    if df is None:
        df = fw
    else:
        df = df.merge(fw, how="left", on="feature")
# use two standard deviations to estimate feature importance
df["weight"] = df.iloc[:, 1:].mean(axis=1) - 2 * df.iloc[:, 1:].std(axis=1)
# take all features that likely have importance (weight >= 0)
df[df["weight"] >= 0]["feature"].to_csv(FEATURES_PATH, index=False)
