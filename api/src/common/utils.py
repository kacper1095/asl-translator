import os


def ensure_dir(path):
    """
    Creates folder with given path, creates parents if necessary
    :param path: Path of folder to create
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def print_info(msg=''):
    """
    Prints formatted info string
    :param msg: Message to print
    :return: None
    """
    print('{0:=^40}'.format(msg.upper()))
