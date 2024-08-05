from PIL import Image as PILImage
import cv2
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
from bananagan.models.pix2pixHD.util import data_utils
from bananagan.models.pix2pixHD.options import TestOptions


class SquarePad:
    """
    SquarePad class to pad an image to make it square
    """
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

def create_label_image(image: PILImage, block_size=13, c=1) -> PILImage:
    """
    create a label image from a PIL image
    :param image:
    :param block_size:
    :param c:
    :return:
    """
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    return PILImage.fromarray(binary)




