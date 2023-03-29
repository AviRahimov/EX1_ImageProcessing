"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import matplotlib as mpl
import cv2
import matplotlib.pyplot as plt
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 214423147

    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    img = cv2.imread(filename)
    if img is None:
        raise Exception("The file you provided isn't exist, pls try different file path")
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img_arr = img.astype(float)
    norm_img_arr = img_arr/255
    return norm_img_arr

    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
def imDisplay(filename: str, representation: int):
    img_arr = imReadAndConvert(filename, representation)

    # Checking if the provided image is a gray image
    if representation == 1:
        plt.imshow(img_arr, cmap='gray')
        plt.show()
    # Checking if the provided image is a RGB image
    elif representation == 2:
        plt.imshow(img_arr)
        plt.show()
    pass

    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    chrom_comp_arr = np.matrix([[0.299, 0.587, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    height = imgRGB[:, :, 0].shape[0]
    weight = imgRGB[:, :, 0].shape[1]
    transposed_imgRGB = imgRGB.reshape(3, -1)
    mat_mult = chrom_comp_arr.dot(transposed_imgRGB)
    print(mat_mult.reshape(3, height, weight))
    # orig_shape = imgRGB.shape
    # imgRGB = imgRGB.reshape(-1, 3)
    # YIQ_img = imgRGB.dot(chrom_comp_arr).reshape(orig_shape)
    # return YIQ_img

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
