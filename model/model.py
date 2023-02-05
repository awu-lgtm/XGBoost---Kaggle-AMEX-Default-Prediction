import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from constants import *
import cudf
import cupy

# given by the organizers
def amex_metric_mod_CPU(y_true, y_pred):

    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


def amex_metric_mod_GPU(y_true, y_pred):

    labels = cupy.transpose(cupy.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = cupy.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[cupy.cumsum(weights) <= int(0.04 * cupy.sum(weights))]
    top_four = cupy.sum(cut_vals[:, 0]) / cupy.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = cupy.transpose(cupy.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = cupy.where(labels[:, 0] == 0, 20, 1)
        weight_random = cupy.cumsum(weight / cupy.sum(weight))
        total_pos = cupy.sum(labels[:, 0] * weight)
        cum_pos_found = cupy.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = cupy.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


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


def get_feature_importances(
    model,
    fold,
    importances,
    metrics=["weight", "gain", "cover", "total_gain", "total_cover"],
):
    for i in range(len(metrics)):
        dd = model.get_score(importance_type=metrics[i])
        df = pd.DataFrame({"feature": dd.keys(), f"importance_{fold}": dd.values()})
        importances[i].append(df)
    return importances


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

train = pd.read_parquet(path=PROCESSED_OUTPUT_PATH)

weights = []
skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# 5 fold KFold CV

# for fold, (train_idx, valid_idx) in enumerate(skf.split(train, train.target)):
#     Xy_train = IterLoadForDMatrix(train.loc[train_idx], FEATURES, "target")

#     X_valid = train.loc[valid_idx, FEATURES]
#     y_valid = train.loc[valid_idx, "target"]

#     dtrain = xgb.DeviceQuantileDMatrix(
#         Xy_train, max_bin=256
#     )  # use DeviceQuantileDMatrix to save memory
#     dvalid = xgb.DMatrix(data=X_valid, label=y_valid)

#     model = xgb.train(
#         xgb_params,
#         dtrain=dtrain,
#         evals=[(dtrain, "train"), (dvalid, "valid")],
#         num_boost_round=9999,
#         early_stopping_rounds=100,
#         verbose_eval=100,
#     )
#     model.save_model(f"XGB_fold{fold}.xgb")

#     importances = get_feature_importances(model, fold, importances)

#     del dtrain, Xy_train, weight
#     del X_valid, y_valid, dvalid, model
#     gc.collect()

# Nested KFold
oof = []


def nested_kfold(train):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    split = outer.split(train, train.target)
    for fold, (train_idx, valid_idx) in enumerate(split):
        X_train, y_train = (
            train.loc[train_idx, FEATURES].to_pandas(),
            train.loc[train_idx, "target"].to_pandas(),
        )

        X_valid = train.loc[valid_idx, FEATURES].to_pandas()
        y_valid = train.loc[valid_idx, "target"].to_pandas()

        model = xgb.XGBClassifier(
            n_estimators=9,
            early_stopping_rounds=100,
            **constant_params,
        )

        rcv = RandomizedSearchCV(
            model,
            param_space,
            n_iter=3,
            scoring="neg_log_loss",
            cv=inner,
            refit=True,
            verbose=4,
        ).fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=100,
        )
        best = rcv.best_estimator_

        best.save_model(f"XGB_fold{fold}.xgb")
        oof_preds = best.predict_proba(X_valid)[:, 1]

        acc = amex_metric_mod_CPU(y_valid.values, oof_preds)
        print("Kaggle Metric = ", acc, "\n")
        tmp = train_gpu.loc[valid_idx, ["customer_ID", "target"]].to_pandas().copy()
        tmp["oof_pred"] = oof_preds
        oof.append(tmp)

        del X_train, y_train, X_valid, y_valid, oof_preds
        del model, rcv, best
        gc.collect()


nested_kfold(train)

# Feature selection
features = None
for importance in importances:
    df = importance[0].copy()
    for k in range(1, FOLDS):
        df = df.merge(importance[k], on="feature", how="left")
    df["importance"] = df.iloc[:, 1:].mean(axis=1)
    df["dif"] = df["importance"] - 1.5 * df.iloc[:, 1:].std(axis=1)
    df = df.sort_values("importance", ascending=False)
    if features is None:
        features = df[df["dif"] >= 0]["feature"]
        print(len(features))
    else:
        features = set(features) & set(df[df["dif"] > 0]["feature"])
        print(len(features))
