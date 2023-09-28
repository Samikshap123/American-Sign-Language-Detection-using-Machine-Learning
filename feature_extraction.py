import cv2
from skimage.feature import hog


def extract_features_hog(frame):
    # Preprocess the frame
    frame = cv2.resize(frame, (64, 128))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the HOG features
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   visualize=False, transform_sqrt=True)

    return features