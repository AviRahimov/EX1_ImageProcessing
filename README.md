# Image Processing Assignment 1
This repository contains several functions for image processing.

## Functions:
1. myID() -> np.int: Returns my ID number.
2. imReadAndConvert(filename: str, representation: int) -> np.ndarray: Reads an image file and converts it to a specified representation (RGB or grayscale).
3. imDisplay(filename: str, representation: int): Reads an image file and displays it as RGB or grayscale.
4. transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray: Converts an RGB image to YIQ color space.
5. transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray: Converts a YIQ image to RGB color space.
6. hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray): Equalizes the histogram of an image.

###Usage
To use these functions, first import the ex1.utils.py file. Then, call the desired function with the appropriate parameters.
### Dependencies
- [x] Python 3.x
- [x] NumPy
- [x] OpenCV
- [x] Matplotlib
- [x] Scikit-learn

### Installation:
Clone the repository:
git clone https://github.com/AviRahimov/EX1_ImageProcessing.git

####install the dependencies:
pip install numpy opencv-python matplotlib scikit-learn
