import numpy as np
import cv2
from PIL import Image

def get_left_images(driving_log):
    labels = driving_log['steering']

    left_images = driving_log['left']
    left_labels = []

    for i in range(len(labels)):
        left_labels.append(np.random.uniform(.15, .3) + labels[i])

    return left_images, np.array(left_labels)

def get_right_images(driving_log):
    labels = driving_log['steering']

    right_images = driving_log['right']
    right_labels = []

    for i in range(len(labels)):
        right_labels.append(labels[i] - np.random.uniform(.15, .3))

    return right_images, np.array(right_labels)

def resize_image(image):
    # width = 160
    # height = 80

    width = 200
    height = 66

    # return cv2.resize(image, (width, height))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def horizontal_flip_image(image, label):
    return cv2.flip(image, 1), -label

def load_image_file(image_path):
    image_path = image_path.strip()
    if image_path[0] != "/":
        image_path = "udacity_data/" + image_path

    image = Image.open(image_path)
    image_array = cv2.imread(image_path)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_array

def load_images(image_paths):
    return list(map(load_image_file, image_paths))
