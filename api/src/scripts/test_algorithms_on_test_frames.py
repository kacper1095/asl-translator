from ..common.config import TrainingConfig, DataConfig, Config
from keras.models import load_model
from ..data_processing.data_generator import prepare_image
from sphinx.versioning import levenshtein_distance

import tqdm
import os
import cv2
import numpy as np


def get_all_models():
    weights = []
    for folder in os.listdir(TrainingConfig.PATHS['MODELS']):
        folder_path = os.path.join(TrainingConfig.PATHS['MODELS'], folder)
        if not os.path.isdir(folder_path):
            continue
        files = os.listdir(folder_path)
        if not 'weights.h5' in files:
            continue
        weights.append((folder, os.path.join(folder_path, 'weights.h5')))
    return weights


def load_data():
    main_path = os.path.join('data', 'test', 'frames')
    data = []
    for folder in sorted(os.listdir(main_path)):
        word = folder.split(' ')[-1]
        images_files = os.listdir(os.path.join(main_path, folder))
        images_files = list(sorted(images_files, key=lambda x: int(x[:-4])))
        images_files = [os.path.join(main_path, folder, img_file) for img_file in images_files]
        data.append((word, images_files))
    return data


def process_data(data, input_shape):
    batch_x = []
    for word, image_files in data:
        for char, img_file in zip(word, image_files):
            img = cv2.imread(img_file).astype(np.float32) / 255.
            if type(input_shape) != tuple:
                continue
            if len(input_shape) == 2:
                img = prepare_image(img)
            elif len(input_shape) == 4:
                img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
                img = img.transpose((2, 0, 1))
            else:
                raise ValueError('Unknown shape: ', input_shape)
            batch_x.append(img)
        yield np.array(batch_x), word
        batch_x.clear()


def test_one_model(model_path):
    model = load_model(model_path, custom_objects={'f1': lambda x, y: x})
    input_shape = model.input_shape
    output_shape = model.output_shape

    if output_shape[-1] == 24:
        DataConfig.use_partial_alphabet()
    else:
        DataConfig.use_full_alphabet()

    data = load_data()
    results = []
    for batch_x, batch_y in process_data(data, input_shape):
        prediction = model.predict(batch_x)
        prediction = np.argmax(prediction, axis=1)
        letters = ''
        for pr in prediction:
            letters += DataConfig.get_letter(pr)[0]
        results.append(levenshtein_distance(letters, ''.join(batch_y)))
    return results


def test():
    results = ['model_name & mean_levensthein \\\\']
    for folder, model_path in tqdm.tqdm(get_all_models()):
        try:
            lev_dist = np.mean(test_one_model(model_path))
            results.append('{0} & {1:.4f} \\\\'.format(folder, lev_dist))
        except Exception as e:
            print('Exception for: {} in {} '.format(model_path, folder))
            print(e)
    with open(os.path.join(TrainingConfig.PATHS['MODELS'], 'result_real_test.txt'), 'w') as f:
        f.write('\n'.join(results))
    return results


if __name__ == '__main__':
    test()
