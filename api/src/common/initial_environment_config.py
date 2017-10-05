import os

# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_BACKEND'] = 'theano'

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
from .utils import print_info


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

print_info("initialized environment")
