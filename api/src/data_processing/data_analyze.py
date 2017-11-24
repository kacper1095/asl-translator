from ..common.config import DataConfig
from collections import defaultdict
from pprint import pprint

import os


def analyze_class_counts():
    train_counter = defaultdict(int)
    test_counter = defaultdict(int)

    def count_under_path(path, counter):
        folders = os.listdir(path)
        for folder in folders:
            counter[folder] += len(os.listdir(os.path.join(path, folder)))
        return counter

    return count_under_path(DataConfig.PATHS['TRAINING_PROCESSED_DATA'], train_counter), count_under_path(
        DataConfig.PATHS['VALID_PROCESSED_DATA'], test_counter)


def main():
    train, test = analyze_class_counts()
    pprint(train)
    pprint(test)


if __name__ == '__main__':
    main()
