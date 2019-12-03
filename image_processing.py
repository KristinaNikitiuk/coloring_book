import cv2
import numpy as np


class ImageProcessing(object):

    def __init__(self, image):
        self.image = image

    def _img_to_gray(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def sobel_filter(self):

        gray_img = self._img_to_gray()

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        grad_x = cv2.filter2D(gray_img, cv2.CV_32F, sobel_x)
        grad_y = cv2.filter2D(gray_img, cv2.CV_32F, sobel_y)

        grad_x_norm = np.abs(grad_x) / np.max(np.abs(grad_x))
        grad_y_norm = np.abs(grad_y) / np.max(np.abs(grad_y))

        result_img = np.sqrt(grad_x_norm ** 2 + grad_y_norm ** 2)
        # theta = np.arctan2(grad_y_norm, grad_x_norm)

        return result_img