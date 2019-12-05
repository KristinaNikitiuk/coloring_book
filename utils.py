import cv2
import matplotlib.pyplot as plt


class Utils(object):

    def __init__(self, image):
        self.image = image

    def image_resize(self, scale_percent):
        width = int(self.image.shape[1] * scale_percent / 100)
        height = int(self.image.shape[0] * scale_percent / 100)
        dsize = (width, height)
        output = cv2.resize(self.image, dsize)
        return output

    def inverte_colors(self):
        return 255 - self.image

    def plot_img(self, img):
        plt.figure(2, figsize=(12, 8))
        plt.subplot(122)
        plt.imshow(img, cmap='gray')
        plt.subplot(121)
        plt.imshow(self.image, cmap='gray')
        plt.show()