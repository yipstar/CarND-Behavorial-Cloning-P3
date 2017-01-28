import numpy as np
import cv2
import math

# resize_height, resize_width = 64, 64
# new_size_col, new_size_row = 64, 64

new_size_col, new_size_row = 200, 66

# new_size_col, new_size_row = 320, 160

# data_dir = "new_data2/"
data_dir = "udacity_data/"

def data_generator(data, pr_threshold, batch_size = 32):

    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)

    for i_batch in range(batch_size):

        print(i_batch)

        i_line = np.random.randint(len(data))
        line_data = data.iloc[[i_line]].reset_index()

        keep_pr = 0

        #x,y = preprocess_image_file_train(line_data)
    
        while keep_pr == 0:
            
            x, y = preprocess_image_from_data(line_data)
            pr_unif = np.random
            
            if abs(y) < .1:
                pr_val = np.random.uniform()
                if pr_val > pr_threshold:
                    keep_pr = 1
            else:
                keep_pr = 1

            #x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            #y = np.array([[y]])

        batch_images[i_batch] = x
        batch_steering[i_batch] = y

        # print("yielding batch_images ", batch_images.shape(0))


    yield batch_images, batch_steering

def preprocess_image_from_data(line_data):

    # select a camera location to use at random
    i_lrc = np.random.randint(3)
    i_lrc = 1

    if (i_lrc == 0):
        path_file = line_data['left'][0].strip()
        shift_ang = .25
    if (i_lrc == 1):
        path_file = line_data['center'][0].strip()
        shift_ang = 0.
    if (i_lrc == 2):
        path_file = line_data['right'][0].strip()
        shift_ang = -.25

    y_steer = line_data['steering'][0] + shift_ang

    # read in image and convert to rgb
    if path_file[0] == "/":
        image = cv2.imread(path_file)
    else:
        image = cv2.imread(data_dir + path_file)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # print(y_steer)
    # print(image.shape)
    # print(image)

    # image, y_steer = translate_image(image, y_steer, 100)
    # image = augment_brightness_camera_images(image)
    image = resize_image(image)

    # flip images at random
    # image = np.array(image)

    # flip a coin to decide whether to flip the image
    ind_flip = np.random.randint(2)
    # ind_flip = 1
    if ind_flip == 0:
        image = cv2.flip(image, 1)
        y_steer = -y_steer

    # print(image)
    # print(image.shape)

    return image, y_steer

# Perform cropping and resize
def resize_image(image):

    shape = image.shape
    # note: numpy arrays are (row, col)!

    # remove top 1/5 of image to remove the horizon (height 160 -> 128)
    # remove bottom 25 pixels to remove the hood (128 - 25 = 103)

    # Disable crop
    # image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]

    # resize the image
    image = cv2.resize(image, (new_size_col, new_size_row), interpolation=cv2.INTER_AREA)

    # perform normalization, do this instead using lambda layer
    #image = image/255.-.5

    return image

def augment_brightness_camera_images(image):

    # convert to HSV
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    random_bright = .25 +np.random.uniform()
    #print(random_bright)

    # randomnly scale the V channel
    image1[:,:,2] = image1[:,:,2] * random_bright

    # convert back to RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

    return image1

# horizontal and vertical shifts
def translate_image(image, steer, trans_range):
    # print(image.shape)
    # return image, steer
    # print("testing")
    # print(steer)

    old_steer = steer

    tr_x = trans_range * np.random.uniform() - trans_range/2
    steer_ang = steer + tr_x/trans_range * 2 * .2

    # steer_ang = old_steer

    # steer_compute = steer + tr_x * 2

    tr_y = 40 * np.random.uniform() - 40/2
    #tr_y = 0

    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    rows, cols, _ = image.shape
    # image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    image_tr = image

    print(image.shape)
    print(image_tr.shape)

    # return image, steer

    return image_tr, steer_ang

# def add_random_shadow(image):
#     top_y = 320*np.random.uniform()
#     top_x = 0
#     bot_x = 160
#     bot_y = 320*np.random.uniform()
#     image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
#     shadow_mask = 0*image_hls[:,:,1]
#     X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
#     Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
# shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
#     #random_bright = .25+.7*np.random.uniform()
#     if np.random.randint(2)==1:
#         random_bright = .5
#         cond1 = shadow_mask==1
#         cond0 = shadow_mask==0
#         if np.random.randint(2)==1:
#             image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
#         else:
#             image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright  
#         image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
#     return image

import pandas as pd
data_dir = "udacity_data"
driving_log = pd.read_csv(data_dir + "/driving_log.csv")

while True:
    next(data_generator(driving_log, 1, 128))
