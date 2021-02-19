import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import xgboost as xgb
import hummingbird.ml

import timeit


X, y = load_iris(return_X_y=True)
X = X.astype(np.float32)  # make sure to use fp32 input


def test_logreg():
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    tvm_model = hummingbird.ml.convert(model, "tvm", X)

    np.testing.assert_equal(model.predict(X), tvm_model.predict(X))


def test_rf():
    model = RandomForestClassifier(max_depth=8)
    model.fit(X, y)

    tvm_model = hummingbird.ml.convert(model, "tvm", X)

    np.testing.assert_equal(model.predict(X), tvm_model.predict(X))


def test_xgb():
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        reg_lambda=1,
        objective="multi:softmax",
        num_class=3,
    )
    model.fit(X, y)

    tvm_model = hummingbird.ml.convert(model, "tvm", X)
    np.testing.assert_equal(model.predict(X), tvm_model.predict(X))


def bench():
    X, y = fetch_california_housing(return_X_y=True)  # input shape: (20640, 8)
    X = X.astype(np.float32)  # make sure to use fp32 input

    model = RandomForestRegressor(max_depth=8, n_estimators=250)
    model.fit(X, y)

    tvm_model = hummingbird.ml.convert(model, "tvm", X)

    loop = 20
    res_sk = timeit.timeit(lambda: model.predict(X), number=loop)
    res_tvm = timeit.timeit(lambda: tvm_model.predict(X), number=loop)

    print(res_sk, res_tvm)


test_logreg()
test_rf()
test_xgb()
bench()
