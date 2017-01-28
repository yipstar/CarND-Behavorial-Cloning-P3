# with open('train.p', 'rb') as f:
#     data = pickle.load(f)

# X_train = data['features']
# y_train = data['labels']

# Load data

# from load_data import *
# X_train, y_train = load_initial_3_frames_of_data_from_driving_log("udacity_data")

# X_data, y_data = load_data_from_driving_log("50hz_data")

# X_data, y_data = load_data_from_driving_log("udacity_data")
# X_data, y_data = load_data_from_driving_log("training_data")

# X2_data, y2_data = load_data_from_driving_log("training_data")

# X_data = np.concatenate((X_data, X2_data), axis=0)
# y_data = np.concatenate((y_data, y2_data), axis=0)

# print("size of data")
# print(X_data.shape)
# print(y_data.shape)

# from angle_smoothing import perform_angle_smoothing
# y_data_smoothed = perform_angle_smoothing(y_data)
# y_data_smoothed = y_data

# print(y_data)
# print(y_data_smoothed)

# from normalize import normalize_greyscale
# X_normalized = normalize_greyscale(X_data)
# X_normalized = X_data

# split into test set
# X_train, X_test, y_train, y_test = train_test_split(
#     X_normalized,
#     y_data_smoothed,
#     test_size=0.05,
#     random_state=832289)

# X_train = X_normalized
# y_train = y_data_smoothed

# # from sklearn.utils import shuffle
# X_train, y_train = shuffle(X_train, y_train)
