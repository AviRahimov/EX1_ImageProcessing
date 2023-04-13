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
# import ex1_utils
import sys

import numpy as np
from ex1_utils import LOAD_GRAY_SCALE
import cv2
from cv2 import createTrackbar

slider_max_val = 100
window_name = "Gamma Correction GUI presentation"


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img

    if img_path is None:
        raise Exception("The file you provided isn't exist, pls try different file path")
    # The image is in grayscale color
    if rep == 1:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # The image is in RGB color
    elif rep == 2:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        raise Exception("You can only write rep as 0 for gray image or 1 for RGB image")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    trackbar_name = f"Gamma "
    createTrackbar(trackbar_name, window_name, 50, slider_max_val, on_trackbar)
    on_trackbar(1)
    cv2.waitKey()
    cv2.destroyAllWindows()
    sys.exit()

def on_trackbar(val):

    # Convert the integer trackbar value to a floating-point gamma value, and set gamma to be 0.001(arbitrary) because
    # if we allow gamma to be 0 we will get an error for dividing by zero
    gamma = val / slider_max_val if val != 0 else 0.001

    # Create a look-up table for gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # The formula assumes that the pixel values in the image are in the range [0, 255], but in reality,the pixel
    # values may be outside this range so, we need to create a look-up table to adjust the pixels in the range [0,255].
    gamma_corrected = cv2.LUT(img, table)

    # Display the gamma corrected image
    cv2.imshow(window_name, gamma_corrected)

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
