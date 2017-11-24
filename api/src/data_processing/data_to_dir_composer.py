import os
import string
import shutil
import glob
import tqdm

from api.src.common.config import DataConfig
from api.src.common.utils import ensure_dir, print_info
from .green_edge_cutter import process_img


AVAILABLE_CHARS = DataConfig.AVAILABLE_CHARS
NUM_OF_CUT_BG_IMAGES_INSTANCES = 100

PATHS = {
    'empslocal': os.path.join(DataConfig.PATHS['RAW_DATA'], 'empslocal', 'dataset5'),
    'massey': os.path.join(DataConfig.PATHS['RAW_DATA'], 'massey', 'images'),
    'handspeak': os.path.join(DataConfig.PATHS['RAW_DATA'], 'handspeak')
}


def parse_empslocal_dataset():
    empslocal = PATHS['empslocal']
    training_volunteers_range = ['A', 'B', 'C', 'D']
    validation_volunteers_range = ['E']
    for volunteer_folder in tqdm.tqdm(os.listdir(empslocal)):
        for letter_folder in os.listdir(os.path.join(empslocal, volunteer_folder)):
            for img_file in os.listdir(os.path.join(empslocal, volunteer_folder, letter_folder)):
                if img_file.startswith('depth'):
                    continue
                src_path = os.path.join(empslocal, volunteer_folder, letter_folder, img_file)
                if letter_folder.lower() not in AVAILABLE_CHARS:
                    continue
                if volunteer_folder in training_volunteers_range:
                    dst_folder_path = os.path.join(DataConfig.PATHS['TRAINING_PROCESSED_DATA'], letter_folder.lower())
                elif volunteer_folder in validation_volunteers_range:
                    dst_folder_path = os.path.join(DataConfig.PATHS['VALID_PROCESSED_DATA'], letter_folder.lower())
                else:
                    raise ValueError("Unknown range for this volunteer: ", volunteer_folder)
                dst_path = os.path.join(dst_folder_path, str(len(os.listdir(dst_folder_path))) + '.png')
                shutil.copy(src_path, dst_path)
    print_info('Processed empslocal')


def parse_massey_data():
    massey = PATHS['massey']
    training_volunteers_range = [1, 2, 3, 4]
    validation_volunteers_range = [5]
    for file_name in tqdm.tqdm(os.listdir(massey)):
        sign, volunteer_number = parse_massey_file_name(file_name)
        src_path = os.path.join(massey, file_name)
        if sign.lower() not in AVAILABLE_CHARS:
            continue
        if volunteer_number in training_volunteers_range:
            dst_folder_path = os.path.join(DataConfig.PATHS['TRAINING_PROCESSED_DATA'], sign.lower())
        elif volunteer_number in validation_volunteers_range:
            dst_folder_path = os.path.join(DataConfig.PATHS['VALID_PROCESSED_DATA'], sign.lower())
        else:
            raise ValueError("Unknown range for this volunteer: ", volunteer_number)
        dst_path = os.path.join(dst_folder_path, str(len(os.listdir(dst_folder_path))) + '.png')
        if volunteer_number in validation_volunteers_range:
            shutil.copy(src_path, dst_path)
        else:
            dst_path = os.path.join(dst_folder_path, str(len(os.listdir(dst_folder_path))) + '.png')
            shutil.copy(src_path, dst_path)
            for _ in range(NUM_OF_CUT_BG_IMAGES_INSTANCES):
                dst_path = os.path.join(dst_folder_path, str(len(os.listdir(dst_folder_path))) + '.png')
                process_img(src_path, dst_path)

    print_info('Processed massey')


def parse_handspeak_data():
    handspeak = PATHS['handspeak']
    for file_name in tqdm.tqdm(os.listdir(handspeak)):
        sign = file_name[:-4]
        src_path = os.path.join(handspeak, file_name)
        if sign.lower() not in AVAILABLE_CHARS:
            continue
        dst_folder = os.path.join(DataConfig.PATHS['TRAINING_PROCESSED_DATA'], sign.lower())
        dst_path = os.path.join(dst_folder, str(len(os.listdir(dst_folder))) + '.png')
        shutil.copy(src_path, dst_path)
        for _ in range(NUM_OF_CUT_BG_IMAGES_INSTANCES):
            dst_path = os.path.join(dst_folder, str(len(os.listdir(dst_folder))) + '.png')
            process_img(src_path, dst_path)
    print_info('Processed heandspeak')


def create_letter_dirs():
    for sign in AVAILABLE_CHARS:
        ensure_dir(os.path.join(DataConfig.PATHS['TRAINING_PROCESSED_DATA'], sign))
        ensure_dir(os.path.join(DataConfig.PATHS['VALID_PROCESSED_DATA'], sign))


def clear_dirs():
    for sign in AVAILABLE_CHARS:
        for file in glob.glob(os.path.join(DataConfig.PATHS['TRAINING_PROCESSED_DATA'], sign, '*')):
            os.remove(file)
        for file in glob.glob(os.path.join(DataConfig.PATHS['VALID_PROCESSED_DATA'], sign, '*')):
            os.remove(file)


def parse_massey_file_name(file_name):
    components = file_name.split('_')
    return components[1], int(components[0][-1])


def main():
    parse_functions = [
        # parse_empslocal_dataset,
        # parse_massey_data,
        parse_handspeak_data
    ]
    create_letter_dirs()
    # clear_dirs()
    for data_function in parse_functions:
        data_function()


if __name__ == '__main__':
    main()
