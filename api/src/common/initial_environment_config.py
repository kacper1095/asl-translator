import os

os.environ['KERAS_BACKEND'] = 'theano'
from .utils import print_info

print_info("initialized environment")
