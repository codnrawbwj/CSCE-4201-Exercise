import cv2
import numpy as np
import os
from keras.models import load_model
from math import atan2, degrees


# Resizing function
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


# Find the slope of glasses
def glasses_slope(p1, p2):
    return degrees(atan2(p2[1] - p1[1], p2[0] - p1[0]))


# Overlay the glasses image on the cat's face
def overlay(background_img, overlaying_image, x_cord, y_cord, overlay_size=None):
    bg_img = background_img.copy()
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        overlaying_image = cv2.resize(overlaying_image.copy(), overlay_size)

    b, g, r, a = cv2.split(overlaying_image)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlaying_image.shape
    roi = bg_img[int(y_cord - h / 2):int(y_cord + h / 2), int(x_cord - w / 2):int(x_cord + w / 2)]

    img1_bg = cv2.bitwise_or(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_bg = cv2.bitwise_and(overlaying_image, overlaying_image, mask=mask)

    bg_img[int(y_cord - h / 2):int(y_cord + h / 2), int(x_cord - w / 2):int(x_cord + w / 2)] = cv2.add(img1_bg, img2_bg)

    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


img_size = 224
# Test with one sample file
path = '/Users/lim/Desktop/CatHisterizer/samples'
filels = sorted(os.listdir(path))

# Read the glasses image
glasses = cv2.imread('/Users/lim/Desktop/CatHisterizer/images/glasses.png', cv2.IMREAD_UNCHANGED)

print('Starting loading models...')

# Bring the models
bbs_model = load_model('/Users/lim/Desktop/CatHisterizer/models/bbs_1.h5')
lmks_model = load_model('/Users/lim/Desktop/CatHisterizer/models/lmks_1.h5')
print('Loading models finished.')

# Start testing
print('Starting testing...')
for f in filels:
    if '.jpg' not in f:
        continue

    img = cv2.imread(os.path.join(path, f))
    ori_img = img.copy()
    result_img = img.copy()

    img, ratio, top, left = resize_img(img)

    inputs = (img.astype('float32') / 255).reshape((1, img_size, img_size, 3))
    pred_bb = bbs_model.predict(inputs)[0].reshape((-1, 2))

    ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(int)

    center = np.mean(ori_bb, axis=0)
    face_size = max(np.abs(ori_bb[1] - ori_bb[0]))
    new_bb = np.array([
        center - face_size * 0.6,
        center + face_size * 0.6
    ]).astype(int)
    new_bb = np.clip(new_bb, 0, 99999)

    face_img = ori_img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]: new_bb[1][0]]
    face_img, face_ratio, face_top, face_left = resize_img(face_img)

    face_inputs = (face_img.astype('float32') / 255).reshape((1, img_size, img_size, 3))

    pred_lmks = lmks_model.predict(face_inputs)[0].reshape((-1, 2))

    new_lmks = ((pred_lmks - np.array([face_left, face_top])) / face_ratio).astype(int)
    ori_lmks = new_lmks + new_bb[0]

    cv2.rectangle(ori_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(255, 255, 255), thickness=2)

    for i, l in enumerate(ori_lmks):
        cv2.putText(ori_img, str(i), tuple(l), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(ori_img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

    glasses_center = np.mean([ori_lmks[0], ori_lmks[1]], axis=0)
    glasses_size = np.linalg.norm(ori_lmks[0] - ori_lmks[1]) * 2

    angle = -glasses_slope(ori_lmks[0], ori_lmks[1])
    M = cv2.getRotationMatrix2D((glasses.shape[1] / 2, glasses.shape[0] / 2), angle, 1)
    rotated_glasses = cv2.warpAffine(glasses, M, (glasses.shape[1], glasses.shape[0]))

    try:
        result_img = overlay(result_img, rotated_glasses, glasses_center[0], glasses_center[1],
                             overlay_size=(int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1])))
    except:
        print('failed overlay image')

    cv2.imshow('img', ori_img)
    cv2.imshow('result', result_img)
    filename, ext = os.path.splitext(f)
    cv2.imwrite('/Users/lim/Desktop/CatHisterizer/result/%s_lmks%s' % (filename, ext), ori_img)
    cv2.imwrite('result/%s_result%s' % (filename, ext), result_img)

    if cv2.waitKey(0) == ord('q'):
        break

print('testing finished.')
