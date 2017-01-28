import pandas as pd
import numpy as np
import cv2
from PIL import Image
from sklearn.utils import shuffle

def data_generator(driving_log, batch_size = 128, p_threshold = 1):

    while 1:

        batch_images = []
        batch_labels = []

        for i in range(batch_size):

            # choose a random log entry
            random_index = np.random.randint(len(driving_log))
            driving_log_entry = driving_log.iloc[[random_index]].reset_index()

            keep_prob = 0
            while keep_prob == 0:

                x, y = augment_log_entry(driving_log_entry)

                if abs(y) < .1:
                    p = np.random.uniform()

                    if p > p_threshold:
                        keep_prob = 1
                else:
                    keep_prob = 1

            batch_images.append(x)
            batch_labels.append(y)

        yield np.array(batch_images), np.array(batch_labels)

def load_data(driving_log):

    images = []
    labels = []

    for i, row in driving_log.iterrows():
        # print(row["center"])

        label = row["steering"]

        # add center images
        image = load_image_file(row["center"])

        # get flipped version before any augmentation
        flipped_image, flipped_label = horizontal_flip_image(image, label)

        # augment and add original center image
        image, label = augment_image(image, label)
        images.append(image)
        labels.append(label)

        # augment and add flipped version
        flipped_image, flipped_label = augment_image(flipped_image, flipped_label)
        images.append(flipped_image)
        labels.append(flipped_label)

        # add left images
        image = load_image_file(row["left"])
        label = label + np.random.uniform(.15, .3)
        image, label = augment_image(image, label)

        images.append(image)
        labels.append(label)

        # add right images
        image = load_image_file(row["right"])
        label = label - np.random.uniform(.15, .3)
        image, label = augment_image(image, label)

        images.append(image)
        labels.append(label)

    images, labels = shuffle(images, labels)

    return np.array(images), np.array(labels)

def augment_image(image, label):
    image = resize_image(image)
    image = adjust_brightness(image)
    return image, label


def augment_log_entry(driving_log_entry):

    label = float(driving_log_entry['steering'][0])
    # print(label)

    random_camera_location = np.random.randint(3)

    if random_camera_location == 0:
        image_path = driving_log_entry['left'][0].strip()
        label = label + np.random.uniform(.15, .3)
    if random_camera_location == 1:
        image_path = driving_log_entry['center'][0].strip()
        label = label + 0.
    if random_camera_location == 2:
        image_path = driving_log_entry['right'][0].strip()
        label = label - np.random.uniform(.15, .3)

    # print(label)

    image = load_image_file(image_path)
    image = resize_image(image)
    image = adjust_brightness(image)
    image, label = translate_image(image, label)

    # flip center images randomly
    if random_camera_location == 1 and np.random.randint(2) == 0:
        image, label = horizontal_flip_image(image, label)

    return image, label

def load_image_file(image_path):
    image_path = image_path.strip()
    if image_path[0] != "/":
        image_path = "udacity_data/" + image_path

    image = Image.open(image_path)
    image_array = cv2.imread(image_path)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_array

def resize_image(image):
    # width = 160
    # height = 80

    width = 200
    height = 66

    # return cv2.resize(image, (width, height))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def horizontal_flip_image(image, label):
    return cv2.flip(image, 1), -label

def translate_image(image, label):
    rows = image.shape[0]
    cols = image.shape[1]

    trans_amount = 100

    x = trans_amount * np.random.uniform() - trans_amount/2
    translated_label = label + x/trans_amount * 2 * .2

    y = 40 * np.random.uniform() - 40/2

    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

    return translated_image, translated_label

def adjust_brightness(image):

    # convert to HSV
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    random_bright = .25 + np.random.uniform()

    # randomnly scale the V channel
    image1[:,:,2] = image1[:,:,2] * random_bright

    # convert back to RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

    return image1



# data_dir = "udacity_data"
# driving_log = pd.read_csv(data_dir + "/driving_log.csv")
# load_data(driving_log)


# while True:
#     x, y = next(data_generator(driving_log, 64))
#     print(x.shape)
#     print(y.shape)
