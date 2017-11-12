import cv2
import os
import datetime
import glob
import numpy as np

from .config import DataConfig, Config
from .utils import ensure_dir


class Logger(object):

    def __init__(self):
        self.initialize_dirs()

    def initialize_dirs(self):
        ensure_dir(DataConfig.PATHS['LOG_DATA_IMAGES'])
        ensure_dir(DataConfig.PATHS['LOG_DATA_TEXT'])

    def log_img(self, img, name=''):
        name = '_' + name if name != '' else name
        if Config.LOGGING:
            self.clear_files_if_necessary()
            if img.max() < 127:
                img = np.copy(img)
                img *= 255
            cv2.imwrite(os.path.join(DataConfig.PATHS['LOG_DATA_IMAGES'], self.get_time_stamp() + name + '.png'), img)

    def log_txt(self, txt):
        if Config.LOGGING:
            self.clear_files_if_necessary()
            with open(os.path.join(DataConfig.PATHS['LOG_DATA_TEXT']), self.get_time_stamp() + '.txt') as f:
                f.write(txt)

    def log(self, msg):
        if Config.LOGGING:
            self.clear_files_if_necessary()
            with open(os.path.join(DataConfig.PATHS['LOG_DATA_TEXT'], 'log.log'), 'a') as f:
                f.write(str(msg) + '\n')

    def clear_files_if_necessary(self):
        files = glob.glob(os.path.join(DataConfig.PATHS['LOG_DATA_IMAGES'], '*'))
        if len(files) > 100:
            os.remove(files[0])

    def get_time_stamp(self):
        timestamp = datetime.datetime.now().strftime('%H_%M_%S_%m_%d')
        return timestamp


logger = Logger()
