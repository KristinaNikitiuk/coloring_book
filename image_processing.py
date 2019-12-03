import cv2

SCALE = 3
DELTA = 0


class ImageProcessing(object):

    def __init__(self, image):
        self.image = image

    def _img_to_gray(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def sobel_filter(self):
        gray_img = self._img_to_gray()
        img = cv2.GaussianBlur(gray_img, (3, 3), 0)

        gradx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
        gradx = cv2.convertScaleAbs(gradx)
        grady = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3, scale=SCALE, delta=DELTA)
        grady = cv2.convertScaleAbs(grady)
        result_img = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)

        return result_img