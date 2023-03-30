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
import cv2
import matplotlib.pyplot as plt
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

chromatic_comp_mat = np.matrix([[0.299, 0.587, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 214423147


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    if img is None:
        raise Exception("The file you provided isn't exist, pls try different file path")
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img_arr = img.astype(float)
    norm_img_arr = img_arr / 255
    return norm_img_arr


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
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


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    imgRGB_mat = imgRGB.reshape(-1, 3)
    mat_mult = np.array(np.dot(imgRGB_mat, chromatic_comp_mat))
    YIQ_mat = mat_mult.reshape(imgRGB.shape)
    return YIQ_mat


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    inverse_chromatic_comp_mat = np.linalg.inv(chromatic_comp_mat)
    imgYIQ_mat = imgYIQ.reshape(-1, 3)
    res_mat = np.array(np.dot(inverse_chromatic_comp_mat, imgYIQ_mat))
    RGB_mat = res_mat.reshape(imgYIQ.shape)
    return RGB_mat


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original Histogram
    :return: (imgEq,histOrg,histEQ)
    """
    # imgOrig is a grayscale image
    if len(imgOrig.shape) == 2:
        imgEq = np.zeros(256)
        norm_imgOrg = (imgOrig/imgOrig.max())
        hist_imgOrg = np.histogram(norm_imgOrg)
        norm_CumSum_imgOrg = (np.cumsum(norm_imgOrg))/(np.cumsum(norm_imgOrg).max())
        LUT = norm_CumSum_imgOrg*255

        for pix in imgOrig:
            loc_arr = np.where(imgOrig == pix)[0]
            imgEq[loc_arr[0]] = LUT[pix]

        hist_imgEq = np.histogram(imgEq)
        return imgEq, hist_imgEq, hist_imgOrg
    # imgOrig is a RGB image
    elif len(imgOrig.shape) == 3:
        YIQ_arr = transformRGB2YIQ(imgOrig)
        Y_channel = YIQ_arr[:, :, 0]

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
