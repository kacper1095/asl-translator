import argparse
import time
import cv2
import numpy as np
import pygame
import pygame.camera as camera
import matplotlib.pyplot as plt
import keras.backend as K


from keras.models import load_model
from ..models.three_convo_change_detection import create_model

all_letters = 'abcdefghiklmnopqrstuvwxy'


def capture_camera(model, model_position_change=None):
    pygame.init()
    camera.init()
    cam = camera.Camera('/dev/video0', (640, 480))
    cam.start()
    ax = plt.subplot(1, 1, 1)
    im = ax.imshow(np.zeros((480, 640, 3)))
    plt.ion()
    previous_frame = None
    while True:
        image = cam.get_image()
        image = np.array(pygame.surfarray.pixels3d(image))
        image = image.transpose((1, 0, 2))
        image = image[:, ::-1]

        h, w, c = image.shape
        x = int(0.7 * w)
        y = 0
        w_window = w - 1
        h_window = int(0.5 * h)
        p1 = (x, y)
        p2 = (w_window, h_window)
        cropped_window = crop_image(image, p1, p2)
        cropped_window = np.array(cropped_window, dtype=np.float32) / 255.
        cropped_window = cv2.resize(cropped_window, (64, 64))
        cropped_window = cropped_window.transpose((2, 0, 1))
        changed_position = False
        if model_position_change is not None and previous_frame is not None:
            changed_position = model_position_change.predict([np.array([previous_frame]), np.array([cropped_window])])[0]
            changed_position = np.argmax(changed_position)
        # if previos_change and not changed_position:
        letter = classify_letter(model, cropped_window)
        last_letter = letter
        previous_frame = cropped_window
        image = image.copy()
        cv2.rectangle(image, p1, p2, (0, 255, 0), 3)
        cv2.putText(image, last_letter, (int(0.8 * w), int(0.8 * h)), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), thickness=6)
        im.set_data(image)
        plt.pause(0.01)
    cam.stop()
    plt.ioff()


def crop_image(image, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return image[y1:y2, x1:x2]


def classify_letter(model, cropped_window):
    prediction = model.predict(np.array([cropped_window]))[0]
    letter_class = int(np.argmax(prediction))
    return all_letters[letter_class]


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def parse_args():
    argparser = argparse.ArgumentParser('Script for testing model on camera')
    argparser.add_argument('model', help='H5 weights of keras model')
    # argparser.add_argument('time', type=int, help='Interval between predictions')
    argparser.add_argument('position_change', default='', help='Test changing position')
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    model = load_model(args.model, custom_objects={'f1': lambda x, y: y})
    model_position_change = None
    if args.position_change != '':
        model_position_change = create_model()
        model_position_change.load_weights(args.position_change)
    # model = args.model
    capture_camera(model, model_position_change)


if __name__ == '__main__':
    main()
