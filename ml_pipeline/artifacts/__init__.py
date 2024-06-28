import time
import os
import pandas as pd
from enum import Enum
from scipy.stats import shapiro, pearsonr
from itertools import product
from pprint import pprint


def line(fonction):
    def wrapper(*args, **kwargs):
        l_ = '=' * 200
        print(l_)
        start = time.time()
        res = fonction(*args, **kwargs)
        print(f"Durée : {round(time.time() - start, 5)} secs")
        return res
    return wrapper


def name_model_path(x_):
    return f'../_models/mlserver/{x_}.joblib'


class MyPaths:
    estimatorFittedStandardEncoder: str = '../_models/artifacts/fitted_standardScaler.joblib'
    estimatorFittedLabelEncoder: str = '../_models/artifacts/fitted_labelEncoder.joblib'
    estimatorFittedOneHotEncoder: str = '../_models/artifacts/fitted_oneHotEncoder.joblib'
    fittedModel: str = f'../_models/mlserver/{round(time.time())}.joblib'