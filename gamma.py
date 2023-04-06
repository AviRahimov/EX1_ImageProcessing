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
import ex1_utils
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

    cv2.namedWindow(window_name)
    trackbar_name = f"Gamma x {slider_max_val}"
    createTrackbar(trackbar_name, window_name, 1, slider_max_val, on_trackbar)
    on_trackbar(1)
    cv2.waitKey()

def on_trackbar(val):
    gamma = val / slider_max_val
    inv_gamma = 1.0 / gamma
    table = (255.0 * (cv2.pow(img / 255.0, inv_gamma))).astype('uint8')  # perform gamma correction on image
    cv2.imshow(window_name, table)

def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
