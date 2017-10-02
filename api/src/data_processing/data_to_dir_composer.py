import os
import string
import shutil
import tqdm

from api.src.common.config import DataConfig
from api.src.common.utils import ensure_dir, print_info


AVAILABLE_CHARS = string.ascii_lowercase + string.digits

PATHS = {
    'empslocal': os.path.join(DataConfig.PATHS['RAW_DATA'], 'empslocal', 'dataset5'),
    'massey': os.path.join(DataConfig.PATHS['RAW_DATA'], 'massey', 'images')
}


def parse_empslocal_dataset():
    empslocal = PATHS['empslocal']
    for volunteer_folder in tqdm.tqdm(os.listdir(empslocal)):
        for letter_folder in os.listdir(os.path.join(empslocal, volunteer_folder)):
            for img_file in os.listdir(os.path.join(empslocal, volunteer_folder, letter_folder)):
                src_path = os.path.join(empslocal, volunteer_folder, letter_folder, img_file)
                dst_folder_path = os.path.join(DataConfig.PATHS['PROCESSED_DATA'], letter_folder.lower())
                dst_path = os.path.join(dst_folder_path, str(len(os.listdir(dst_folder_path))) + '.png')
                shutil.copy(src_path, dst_path)
    print_info('Processed empslocal')


def parse_massey_data():
    massey = PATHS['massey']
    for file_name in tqdm.tqdm(os.listdir(massey)):
        sign = parse_massey_file_name(file_name)
        src_path = os.path.join(massey, file_name)
        dst_folder_path = os.path.join(DataConfig.PATHS['PROCESSED_DATA'], sign.lower())
        dst_path = os.path.join(dst_folder_path, str(len(os.listdir(dst_folder_path))) + '.png')
        shutil.copy(src_path, dst_path)
    print_info('Processed massey')

DATASET_FUNCTIONS = [
    parse_empslocal_dataset,
    parse_massey_data
]


def create_letter_dirs():
    for sign in AVAILABLE_CHARS:
        ensure_dir(os.path.join(DataConfig.PATHS['PROCESSED_DATA'], sign))


def clear_dirs():
    for sign in AVAILABLE_CHARS:
        shutil.rmtree(os.path.join(DataConfig.PATHS['PROCESSED_DATA', sign, '*']))


def parse_massey_file_name(file_name):
    return file_name.split('_')[1]


def main(args):
    create_letter_dirs()
    clear_dirs()
    for data_function in DATASET_FUNCTIONS:
        data_function()


if __name__ == '__main__':
    main(None)
