import argparse
import time
import cv2
import numpy as np
import pygame
import pygame.camera as camera
import matplotlib.pyplot as plt


from keras.models import load_model

all_letters = 'abcdefghiklmnopqrstuvwxy'


def capture_camera(model, time_interval):
    pygame.init()
    camera.init()
    cam = camera.Camera('/dev/video0', (640, 480))
    cam.start()
    ax = plt.subplot(1, 1, 1)
    im = ax.imshow(np.zeros((480, 640, 3)))
    plt.ion()
    start = time.time()
    last_letter = ''
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
        if time.time() - start > time_interval:
            letter = classify_letter(model, cropped_window)
            last_letter = letter
            start = time.time()
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
    cropped_window = np.array(cropped_window, dtype=np.float32) / 255.
    cropped_window = cv2.resize(cropped_window, (64, 64))
    cropped_window = cropped_window.transpose((2, 0, 1))
    prediction = model.predict(np.array([cropped_window]))[0]
    letter_class = int(np.argmax(prediction))
    return all_letters[letter_class]


def parse_args():
    argparser = argparse.ArgumentParser('Script for testing model on camera')
    argparser.add_argument('model', help='H5 weights of keras model')
    argparser.add_argument('time', type=int, help='Interval between predictions')
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    model = load_model(args.model, custom_objects={'f1': lambda x, y: y})
    # model = args.model
    capture_camera(model, args.time)


if __name__ == '__main__':
    main()
