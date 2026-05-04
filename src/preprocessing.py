import cv2
import numpy as np
import pandas as pd
import os

def apply_clahe(img):
    """
    Improves local contrast for images. Helps with dark or washed out signs.
    CLAHE works in the LAB color space because it only affects the L channel (brightness) and not the A and B channels (colors)
    """

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def load_and_preprocess(img_path, img_size=32, use_clahe=False):
    """
    loads a single image and runs the preprocessing pipeline.
    returns a float32 numpy array of shape (32, 32, 3)
    """

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    if use_clahe:
        img = apply_clahe(img)

    img = img.astype(np.float32)/255.0
    return img

def load_dataset(data_dir, img_size=32, use_clahe=False):
    """
    goes through the training data and loads every image + its label
    Label is the folder name (0-42) cast to an int
    returns X as a float32 array of shape (N, 32, 32, 3) and y as an int array of shape (N,)
    """
    X, y = [], []

    class_ids = sorted(os.listdir(data_dir))
    for class_id in class_ids:
        class_path = os.path.join(data_dir, class_id)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = load_and_preprocess(img_path, img_size, use_clahe)
            X.append(img)
            y.append(int(class_id))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_test_dataset(test_csv_path, data_dir, img_size=32, use_clahe=False):
    """
    Loads test set using the csv file for labels
    note for this dataset the test data is not organized into class folders
    """

    df = pd.read_csv(test_csv_path)
    X, y = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, row['Path'])
        img = load_and_preprocess(img_path, img_size, use_clahe)
        X.append(img)
        y.append(int(row['ClassId']))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)