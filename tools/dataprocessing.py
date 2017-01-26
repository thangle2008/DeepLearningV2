import os
import re
import multiprocessing

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.misc import imread, imresize

def load_img(img_path, class_id, target_size, transpose=False):
    """
    Loads the image and resizes it to target_size.
    The class_id is passed along to preserve the label of the image.
    """

    img = imread(img_path)
    img = imresize(img, target_size)

    # change from (width, height, channels) to (channels, width, height)
    if transpose:
        img = img.transpose((2, 0, 1))

    return img, class_id


def split_data(folder, target_size, train=.6, transpose=False, seed=42):
    """
    Loads the data in the specified folder and splits them into training,
    validation and test sets. The amount of data in each dataset 
    is determined by the proportion. The folder should have different 
    subdirectory for each category. 
    Also returns the following information: training, 
    validation and test sets (as numpy arrays) as well as the number 
    of categories.
    """

    data = []
    categories = []
    num_classes = 0

    # get the file paths
    for root, dirnames, filenames in os.walk(folder):  
        dirnames.sort()
        if root == folder:
            continue
        current_dir = root.split('/')[-1]

        print current_dir
        for filename in filenames:
            if re.search('\.(jpg|png|jpeg)$', filename):
                filepath = os.path.join(root, filename)
                data.append(filepath)
                categories.append(num_classes)
        num_classes += 1

    # load images asynchronically 
    pool = multiprocessing.Pool()
    
    results = [pool.apply_async(load_img, (data[i], categories[i], target_size,
            transpose)) for i in range(len(data))]
    loaded_images = [r.get() for r in results]

    data, categories = zip(*loaded_images)
    data = np.asarray(data)
    categories = np.array(categories, dtype=np.int32)

    # split data in stratified fashion
    train_x, val_x, train_y, val_y = train_test_split(data, categories, 
                                            train_size=train, stratify=categories)

    return (train_x, train_y), (val_x, val_y), num_classes
