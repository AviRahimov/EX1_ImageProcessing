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
import math
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
# from sklearn.cluster import KMeans

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


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    if len(imgRGB.shape) == 2:
        return
    # Use np.einsum() to perform matrix multiplication without flattening and reshaping
    imgYIQ = np.einsum('ijk,lk->ijl', imgRGB, chromatic_comp_mat)
    # Display the output image
    # Return the output image
    return imgYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # Define the matrix for converting YIQ to RGB
    inverse_chromatic_comp_mat = np.linalg.inv(chromatic_comp_mat)
    # Get the shape of the input image
    imgRGB = np.einsum('ijk,lk->ijl', imgYIQ, inverse_chromatic_comp_mat)
    # Display the output image
    # Return the output image
    return imgRGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original Histogram
    :return: (imgEq,histOrg,histEQ)
    """
    Origimg = Color_Or_Gray_Image(imgOrig)
    YIQimg = transformRGB2YIQ(imgOrig)

    # stretching the original image because the original image is between the range [0,1]
    stretched_imgOrg = cv2.normalize(Origimg, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # calculating the histogram of the original image with 256 bins and take the first argument([0]) that is the data
    hist_imgOrg = np.histogram(stretched_imgOrg, bins=256)[0]
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
    # imgEq = imgEq/imgEq.max()
    hist_imgEq = np.histogram(imgEq, bins=256)[0]
    # if the original image was colored so we need to turn it back to RGB as we worked on the Y channel
    imgEq = imgEq / imgEq.max()
    if len(imgOrig) == 3:
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
    # List of the quantized image in each iteration.
    qImage = []
    # List of the MSE error in each iteration.
    error = []
    # The borders which divide the histograms into segments, size of z = nQuant+1.
    z = []
    Origimg = Color_Or_Gray_Image(imOrig)
    YIQimg = transformRGB2YIQ(imOrig)

    # stretching the original image because the original image is between the range [0,1]
    stretched_imgOrg = cv2.normalize(Origimg, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    copy_img = stretched_imgOrg.copy()
    imgOrig_hist = np.histogram(stretched_imgOrg.flatten(), bins=256)[0]
    # Computing the cumulative sum for the image for finding the first boundaries
    imgCumSum = np.cumsum(imgOrig_hist)
    z.append(0)
    # first boundary will be the number of pixels dividing by nQuant as writing in assignment PDF that we need to set
    # initial segment division
    Quant = imgCumSum[-1] / nQuant
    # setting the rest of the boundaries
    for boundary in range(1, nQuant):
        z.append(np.where(imgCumSum >= boundary * Quant)[0][0])
    z.append(255)

    for iteration in range(nIter):
        # The values that each of the segments’ intensities will map to.
        q = []
        # calculating mean for each segment(q)
        for i in range(nQuant):
            mean = np.mean(stretched_imgOrg[(stretched_imgOrg > z[i]) & (stretched_imgOrg <= z[i + 1])])
            q.append(mean)

        # Create the quantize image
        Quantimg = np.zeros_like(copy_img)
        for i in range(nQuant):
            Quantimg[(stretched_imgOrg > z[i]) & (stretched_imgOrg <= z[i + 1])] = q[i]

        # Calculating new values(fixed values) for z
        z.clear()
        z.append(0)
        for i in range(len(q) - 1):
            z.append(int((q[i] + q[i + 1]) / 2))
        z.append(255)
        MSE_error = MSE(Origimg * 255, Quantimg)
        error.append(MSE_error)
        if iteration > 5:
            if check_MSE_convergence(error, math.pow(10, -5)):
                break
        qImage.append(Quantimg / 255.0)
    # if the image is colored so we need to convert it back to RGB since we worked on the Y channel
    if len(imOrig) == 3:
        for i in range(len(qImage)):
            YIQimg[:, :, 0] = qImage[i]
            qImage[i] = transformYIQ2RGB(YIQimg)

    return qImage, error


def check_MSE_convergence(MSE_list: list, tolerance: float) -> bool:
    # checking if the difference between the last 10 numbers is less than 10^-5
    for i in range(-6, -1):
        if (MSE_list[i] - MSE_list[i + 1]) > tolerance:
            return False
    return True

def Color_Or_Gray_Image(img: np.ndarray) -> np.ndarray:
    img_len = len(img.shape)

    # imgOrig is a RGB image
    if img_len == 3:
        # transform the image to YIQ image and taking only the Y channel.
        return transformRGB2YIQ(img)[:, :, 0]
    # imgOrig is a grayscale image
    elif img_len == 2:
        # if the image is already RGB image so we leave it like that
        return img

# Kmeans implementation for image quantization
# def quantizeImage(img, num_colors, num_iterations):
#     # Flatten the image into a 2D array of pixels
#     pixels = img.reshape((-1, 3))
#
#     # Perform K-means clustering to determine the centroids of the color clusters
#     kmeans = KMeans(n_clusters=num_colors, max_iter=num_iterations).fit(pixels)
#     centroid_colors = kmeans.cluster_centers_
#
#     # Quantize the image by replacing each pixel with the nearest centroid color
#     quantized_pixels = np.zeros_like(pixels)
#     for i in range(num_colors):
#         quantized_pixels[kmeans.labels_ == i] = centroid_colors[i]
#
#     # Reshape the quantized pixel array back into an image
#     quantized_img = quantized_pixels.reshape(img.shape)
#
#     # Compute the mean squared error between the original and quantized images
#     mse = np.mean((img - quantized_img) ** 2)
#
#     # Return the quantized image and the MSE
#     return [quantized_img], [mse]
