import cv2

from image_processing import ImageProcessing
from utils import Utils


SCALE_PERCENT = 30


if __name__ == '__main__':

    src = cv2.imread('images/peppa.png', 0)

    # output = Utils(src).image_resize(SCALE_PERCENT)
    countoured_img = ImageProcessing(src).sobel_filter()
    res = Utils(countoured_img).inverte_colors()
    Utils(src).plot_img(res)

