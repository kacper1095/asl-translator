from api.src.keras_extensions import metrics
from keras import backend as K

import numpy as np

metrics_to_test = [
    metrics.f1,
    metrics.precision,
    metrics.recall
]


def test_shape_metrics():
    y_true = K.variable(np.random.randint(0, 1, size=(100, 24)))
    y_pred = K.variable(np.random.randint(0, 1, size=(100, 24)))
    for metric in metrics_to_test:
        output = metric(y_true, y_pred)
        assert K.eval(output).shape == ()


def test_values():
    y_true = K.variable(np.asarray([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]))

    y_pred = K.variable(np.asarray([
        [0.1, 0.8, 0.5, 0.5],
        [0, 1.0, 0, 0],
        [0.1, 0.1, 0.1, 0.7]
    ]))

    assert_values = [0.6666659116744995, 0.6666666865348816, 0.6666666865348816]
    for metric, correct_value in zip(metrics_to_test, assert_values):
        output = metric(y_true, y_pred)
        assert K.eval(output) - correct_value < 1e-5


