import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_info(msg=''):
    print('{0:=^40}'.format(msg.upper()))
