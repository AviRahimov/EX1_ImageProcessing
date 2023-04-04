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
    res_mat = np.array(np.dot(imgYIQ_mat, inverse_chromatic_comp_mat))
    RGB_mat = res_mat.reshape(imgYIQ.shape)
    return RGB_mat


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original Histogram
    :return: (imgEq,histOrg,histEQ)
    """
    Origimg = 0
    YIQimg = 0
    img_len = len(imgOrig.shape)
    # imgOrig is a RGB image
    if img_len == 3:
        # transform the image to YIQ image and taking only the Y channel
        YIQimg = transformRGB2YIQ(imgOrig)
        Origimg = YIQimg[:, :, 0]
    # imgOrig is a grayscale image
    elif img_len == 2:
        # if the image is already RGB image so we leave it like that
        Origimg = imgOrig

    # stretching the original image because the original image is between the range [0,1]
    # stretched_imgOrg = np.ndarray(Origimg*255).astype('uint8')
    stretched_imgOrg = cv2.normalize(Origimg, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # calculating the histogram of the original image with 256 bins and take the first argument([0]) that is the data
    hist_imgOrg = np.histogram(stretched_imgOrg.flatten(), bins=256)[0]
    # calculating the cumulative sum of the original image and normalize it
    CumSum_imgOrig = np.cumsum(hist_imgOrg)
    norm_CumSum_imgOrg = CumSum_imgOrig / (CumSum_imgOrig.max())
    # calculating the look up table for each intensity in the normalized cumulative sum and multiply it by 255
    # so the range be between [0,255]
    LUT = np.ceil(norm_CumSum_imgOrg * 255)
    # changing the old colors by the new colors in the look up table
    imgEq = stretched_imgOrg.copy()
    for color in range(256):
        imgEq[stretched_imgOrg == color] = LUT[color]
    # normalize the equalized image and create histogram for it
    imgEq = imgEq/imgEq.max()
    hist_imgEq = np.histogram(imgEq.flatten(), bins=256)[0]
    # if the original image was colored so we need to turn it back to RGB as we worked on the Y channel
    if img_len == 3:
        YIQimg[:, :, 0] = imgEq
        imgEq = transformYIQ2RGB(YIQimg)
    return imgEq, hist_imgOrg, hist_imgEq

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
    Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])
    """
    pass
