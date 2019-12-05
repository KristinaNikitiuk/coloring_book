import cv2
import numpy as np

SCALE = 2
DELTA = 0


class ImageProcessing(object):

    def __init__(self, image):
        self.image = image

    def _remove_shadow(self, img):
        rgb_planes = cv2.split(img)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((3, 3), np.uint16))
            bg_img = cv2.medianBlur(dilated_img, 7)
            norm_img = cv2.normalize(bg_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            _, thr_img = cv2.threshold(norm_img, 255, 0, cv2.THRESH_TRUNC)
            normalize = cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            result_planes.append(bg_img)
            result_norm_planes.append(normalize)

        result_norm = cv2.merge(result_norm_planes)
        return result_norm

    def sobel_filter(self):
        img = self._remove_shadow(self.image)

        gradx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, scale=SCALE, delta=DELTA)
        gradx = cv2.convertScaleAbs(gradx)
        grady = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, scale=SCALE, delta=DELTA)
        grady = cv2.convertScaleAbs(grady)
        result_img = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)

        return result_img