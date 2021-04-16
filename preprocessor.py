from tensorflow.keras.preprocessing import image
from cv2 import cv2
import numpy as np

img_height = 576
img_width = 576

target_height = 144
target_width = 144

def center_crop(img):
    h = img.shape[0]
    w = img.shape[1]
    left = int(w/4)
    top = int(h/4)
    right = left + int(w/2)
    bottom = top + int(h/2)

    img = img[top:bottom, left:right,]
    return img

def denoise(img):
    return cv2.fastNlMeansDenoising(img,None,10,7,21)

def load_image(path, rotate=True):
    img = cv2.imread(path,0)
    img = denoise(img)
    cv2.imwrite('temp.png', img)
    img = image.load_img('temp.png', target_size = (img_height, img_width), color_mode='grayscale')
    img = image.img_to_array(img)
    img = center_crop(center_crop(img))
    img = img/255.0
    return img
