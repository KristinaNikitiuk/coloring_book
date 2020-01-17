import cv2

from coloring_book.image_processing import ImageProcessing
from coloring_book.utils import Utils

SCALE_PERCENT = 30


def main():
    src = cv2.imread('images/peppa1.png', 0)

    countoured_img = ImageProcessing(src).sobel_filter()
    res = Utils(countoured_img).inverte_colors()
    Utils(src).plot_img(res)


if __name__ == '__main__':
    main()

