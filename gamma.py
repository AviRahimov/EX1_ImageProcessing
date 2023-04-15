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
import sys
import numpy as np
from ex1_utils import imReadAndConvert
from ex1_utils import LOAD_GRAY_SCALE
import cv2
from cv2 import createTrackbar

slider_max_val = 200
window_name = "Gamma Correction GUI presentation"


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    img = imReadAndConvert(img_path, rep)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    trackbar_name = f"Gamma "
    createTrackbar(trackbar_name, window_name, 50, slider_max_val, on_trackbar)
    cv2.waitKey()
    cv2.destroyAllWindows()
    sys.exit()

def on_trackbar(val):
    gamma = val / (slider_max_val/2)
    gamma_corrected = np.power(img, gamma)
    cv2.imshow(window_name, gamma_corrected)

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
