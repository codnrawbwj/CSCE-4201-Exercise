import random
import pandas
import numpy
import cv2
import os

# Fixed input size
img_size = 224


# Unify the sizes of all input images
def resize_img(im):
    old_size = im.shape[:2]
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return new_im, ratio, top, left


# There are 7 directories of dataset
# Reading the input images
for i in range(7):
    if i == 0:
        print("preprocessing 1st dataset")
    elif i == 1:
        print("preprocessing 2nd dataset")
    elif i == 2:
        print("preprocessing 3rd dataset")
    else:
        print("preprocessing " + str(i + 1) + "th dataset")
    dirname = 'CAT_0' + str(i)
    path = '/Users/lim/Desktop/CatHisterizer/cats/%s' % dirname
    filels = sorted(os.listdir(path))
    random.shuffle(filels)

    # Store in 3 different fields
    dataset = {
        'imgs': [],  # images
        'lmks': [],  # landmarks
        'bbs': []  # bounding boxes
    }

    for f in filels:
        # Only read the landmark coordinate files
        if '.cat' not in f:
            continue

        # Resize the images
        pd_frame = pandas.read_csv(os.path.join(path, f), sep=' ', header=None)
        # Error detected: using 'as_matrix' instead of 'values', 'as_matrix' is not used anymore.
        landmarks = (pd_frame.values[0][1:-1]).reshape((-1, 2))

        img_filename, ext = os.path.splitext(f)

        img = cv2.imread(os.path.join(path, img_filename))

        img, ratio, top, left = resize_img(img)
        landmarks = ((landmarks * ratio) + numpy.array([left, top])).astype(int)
        bb = numpy.array([numpy.min(landmarks, axis=0), numpy.max(landmarks, axis=0)])

        dataset['imgs'].append(img)
        dataset['lmks'].append(landmarks.flatten())
        dataset['bbs'].append(bb.flatten())

    # Save as npy files
    numpy.save('/Users/lim/Desktop/CatHisterizer/dataset/%s.npy' % dirname, numpy.array(dataset))
