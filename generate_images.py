import pandas as pd

data_dir = "new_data"
driving_log = pd.read_csv(data_dir + "/driving_log.csv")

from generator import data_generator

datagen = data_generator(driving_log, 1, 10)

X, y = next(datagen)

print(X.shape)
print(y.shape)

# from PIL import Image

# for image in X:

# im = Image.fromarray(A)
# im.save("your_file.jpeg")
