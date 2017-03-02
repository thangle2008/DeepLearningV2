import os
import re
import multiprocessing
import cPickle
import random

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.misc import imread, imresize, imrotate, toimage, fromimage
import PIL
from PIL.ImageEnhance import Brightness, Contrast, Color 

def _load_img(img_path, class_id, target_size, transpose=False):
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


#### Color jittering functions
def _adjust(img, alpha, etype):
    if alpha == 0.0:
        return img

    pil_img = toimage(img)

    enhancer = None
    if etype == "brightness":
        enhancer = Brightness(pil_img)
    elif etype == "color":
        enhancer = Color(pil_img)
    elif etype == "contrast":
        enhancer = Contrast(pil_img)

    return fromimage(enhancer.enhance(alpha))


def color_jitter(img):
    """
    Randomly adjust brightness, contrast and color of the image.
    """
    etypes = ['brightness', 'contrast', 'color']

    # Random the order of enhancement 
    random.shuffle(etypes)

    # Adjust consecutively
    new_img = np.array(img)
    for e in etypes:
        alpha = random.uniform(0.5, 1.5)
        new_img = _adjust(new_img, alpha, e)

    return new_img


def split_data(folder, target_size, train=.6, transpose=False):
    """
    Loads the data in the specified folder and splits them into training 
    and test sets. The amount of data in each dataset 
    is determined by the proportion. The folder should have different 
    subdirectory for each category. 
    Returns the following information: training, and test sets 
    (as numpy arrays) as well as the mapping from num to class name.
    """

    data = []
    categories = []
    num_classes = 0
    class_map = dict()

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

        class_map[num_classes] = current_dir
        num_classes += 1

    # load images asynchronically 
    pool = multiprocessing.Pool()
    
    results = [pool.apply_async(_load_img, (data[i], categories[i], target_size,
            transpose)) for i in range(len(data))]
    loaded_images = [r.get() for r in results]

    data, categories = zip(*loaded_images)
    data = np.asarray(data)
    categories = np.array(categories, dtype=np.int32)

    print "Number of samples =", data.shape
    print class_map

    if train == 1.0:
        return (data, categories), None, class_map

    # split data in stratified fashion
    X_train, y_train, X_test, y_test = None, None, None, None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=train)

    train_index, test_index = next(iter(sss.split(data, categories)))
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = categories[train_index], categories[test_index]

    return (X_train, y_train), (X_test, y_test), class_map


def get_pickle(indir, outfile, target_size, train=0.6, transpose=False):
    """
    Load images in a folder and output using pickle.
    """

    train_data, test_data, class_map = split_data(indir, target_size, 
                                            train=train, transpose=transpose)

    # write the data with smaller size first (assume test_data)
    write_list = [class_map, test_data, train_data]

    with open(outfile, 'wb') as op:
        for data in write_list:
            cPickle.dump(data, op)