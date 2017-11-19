import os
os.environ['THEANO_FLAGS'] = 'device=cpu'

from ..common import initial_environment_config
from ..common.config import TrainingConfig

import glob
import numpy as np
import tqdm
import time

from keras.models import load_model

TESTING_BATCH_SIZES = [1, 2, 4, 8, 16]
NB_TRIES = 10
INPUT_DIMS = (3, 64, 64)


def test():
    for weight_file in tqdm.tqdm(list_model_folders()):
        try:
            test_speed(weight_file)
        except Exception as e:
            print(e)


def list_model_folders():
    return glob.glob(os.path.join(TrainingConfig.PATHS['MODELS'], '**', '*.h5'))


def test_speed(model_path):
    filename = 'speedtest.txt'
    path = os.path.dirname(model_path)
    if os.path.exists(os.path.join(path, filename)):
        return
    model = load_model(model_path, custom_objects={'f1': lambda x, y: x})
    speed_test_text = get_speed_results(model)
    with open(os.path.join(path, filename), 'w') as f:
        f.write(speed_test_text)


def get_speed_results(model):
    text_lines = ['batch_size & time_mean & time_std \\\\']
    for batch_size in TESTING_BATCH_SIZES:
        input_shape = (batch_size,) + tuple(model.input_shape[1:])
        times = []
        initiate_computational_graph_for(model, input_shape)
        for _ in range(NB_TRIES):
            x = np.random.normal(size=input_shape)
            start = time.time()
            model.predict(x, batch_size)
            end = time.time()
            times.append(end - start)
        text_lines.append('{} & {} & {} \\\\'.format(batch_size, np.mean(times), np.std(times)))
    return '\n'.join(text_lines)


def initiate_computational_graph_for(model, input_shape):
    model.predict(np.random.normal(size=input_shape))

if __name__ == '__main__':
    test()
