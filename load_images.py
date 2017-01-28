import pandas as pd
from sklearn.utils import shuffle
from preprocessing import *

def load_data(data_dir):

    driving_log = pd.read_csv(data_dir + "/driving_log.csv")

    # list
    images = load_images(driving_log["center"])

    # ndarray
    labels = driving_log["steering"]

    left_images, left_labels = get_left_images(driving_log)
    right_images, right_labels = get_right_images(driving_log)

    left_images = load_images(left_images)
    right_images = load_images(right_images)


    images = images + left_images + right_images
    labels = np.concatenate([labels, left_labels, right_labels])

    flipped_images = []
    flipped_labels = []

    for i in range(len(images)):
        flipped_image, flipped_label = horizontal_flip_image(images[i], labels[i])
        flipped_images.append(flipped_image)
        flipped_labels.append(flipped_label)

    full_images = images + flipped_images
    full_labels = np.concatenate([labels, np.array(flipped_labels)])

    full_images = list(map(resize_image, full_images))

    print("size of test data: ", len(full_images))

    full_images = np.array(full_images)

    full_images, full_labels = shuffle(full_images, full_labels)

    return full_images, full_labels


