import pickle
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from imutils import paths

# get all the image paths
image_paths = list(paths.list_images('.\\input\\preprocessed_image'))

# create a DataFrame 
data = pd.DataFrame()

labels = []
for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
    label = image_path.split(os.path.sep)[-2]
    # save the relative path for mapping image to target
    data.loc[i, 'image_path'] = image_path

    labels.append(label)

# labels = np.array(labels)
labels = np.hstack(labels)
# one hot encode the labels
lb = LabelEncoder()
labels = lb.fit_transform(labels)

print(labels)
print(f"The first one hot encoded labels: {labels[0]}")
print(f"Mapping the first one hot encoded label to its category: {lb.classes_[1]}")
print(f"Total instances: {len(labels)}")

data['target'] = labels

# for i in range(len(labels)):
#     index = np.argmax(labels[i])
#     data.loc[i, 'target'] = int(index)

# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# save as CSV file
data.to_csv('./input/data.csv', index=False)

# pickle the binarized labels
print('Saving the binarized labels as pickled file')
with open('labels.pickle', 'wb') as f:
    pickle.dump(labels, f)
print("Saved")
print(data.head(10))