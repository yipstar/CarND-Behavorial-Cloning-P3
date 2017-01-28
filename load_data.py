import pandas as pd
from PIL import Image
import numpy as np

def load_data_from_driving_log(data_dir):
    driving_log = pd.read_csv(data_dir + "/driving_log.csv")
    return load_data_from_df(driving_log, data_dir)

def load_initial_3_frames_of_data_from_driving_log(data_dir):
    driving_log = pd.read_csv(data_dir + "/driving_log.csv")

    straight = driving_log[driving_log["steering"] == 0.0]
    positive = driving_log[driving_log["steering"] > 0.0]
    negative = driving_log[driving_log["steering"] < 0.0]

    # print("number of straight steering examples, ", len(straight))
    # print("number of positive steering examples, ", len(positive))
    # print("number of negative steering examples, ", len(negative))

    first_straight = straight.iloc[0]
    first_positive = positive.iloc[0]
    first_negative = negative.iloc[0]

    frames = [first_straight, first_positive, first_negative,
              straight.iloc[1], positive.iloc[1], negative.iloc[1]
    ]

    df = pd.DataFrame(frames)

    return load_data_from_df(df, data_dir)

def load_data_from_df(df, data_dir):
    # straight = df[df["steering"] == 0.0]

    # # positive = df[df["steering"] > 0.0]
    # # negative = df[df["steering"] < 0.0]

    # non_straight = df[df["steering"] != 0.0]

    # print("number of straight steering examples, ", len(straight))
    # print("number of non-straight steering examples, ", len(non_straight))

    # print("number of positive steering examples, ", len(positive))
    # print("number of negative steering examples, ", len(negative))

    nonZeroSamples =  df.loc[df['steering'] != 0.0]
    zeroSamples =  df.loc[df['steering'] == 0.0]

    print("number of non zero steering examples, ", len(nonZeroSamples))
    print("number of zero steering examples, ", len(zeroSamples))

    features = []
    labels = []

    for index, row in df.iterrows():

        image_path = row["center"]
        if image_path[0] == "/":
            parts = image_path.split("/")
            image_path = parts[-2] + "/" + parts[-1]

        image = Image.open(data_dir + "/" + image_path)
        image.load()

        # Load image data as 1 dimensional array
        # We're using float32 to save on memory space
        # feature = np.array(image, dtype=np.float32).flatten()
        feature = np.array(image, dtype=np.float32)

        # print(feature)

        features.append(feature)

        label = row["steering"]
        # print(label)

        labels.append(label)

    X_data = np.array(features)
    y_data = np.array(labels)

    return (X_data, y_data)
