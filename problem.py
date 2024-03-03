import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import rampwf as rw

problem_title = "Olympic Medal Prediction"
Predictions = rw.prediction_types.make_multiclass(label_names=['Gold', 'Silver', 'Bronze', 'No Medal'])
workflow = rw.workflows.FeatureExtractorClassifier()

score_types = [
    rw.score_types.BalancedAccuracy(
        name="bal_acc", precision=3, adjusted=False
    ),
    rw.score_types.Accuracy(name="acc", precision=3),
]

target_column_name = 'Medal'

def get_data(path='.', split='train'):
    data_path = os.path.join(path, 'data', f'{split}.csv')
    data = pd.read_csv(data_path)
    y = data[target_column_name].map({1: 'Gold', 2: 'Silver', 3: 'Bronze', 4: 'No Medal'})
    X = data.drop(columns=[target_column_name])
    return X, y

def get_train_data(path='.'):
    return get_data(path, 'train')

def get_test_data(path='.'):
    return get_data(path, 'test')

def get_cv(X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return cv.split(X, y)
