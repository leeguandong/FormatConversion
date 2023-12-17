'''
@Time    : 2022/1/6 16:48
@Author  : leeguandon@gmail.com
'''
import os
import torch
import argparse
import cv2
import pylab
import numpy as np
import typing
import fitz  # pip install PyMuPDF https://pypi.org/project/PyMuPDF/ 不是 pip install fitz
import glob
from scipy import misc
from PIL import Image, ImageCms
# from libtiff import TIFF
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

torch.manual_seed(2022)


def isFloat(img) -> bool:
    """
    Decide whether image pixels or an individual color is floating point or byte

    :param img: can either one be a numpy array, or an image

    :return bool:
    """
    if isinstance(img, np.ndarray):
        if len(img.shape) < 2:  # a single color
            if img.shape or isinstance(img[0], (np.float, np.float64)):
                return True
        elif len(img.shape) < 3:  # black and white is [x,y,val] not [x,y,[val]]
            if isinstance(img[0, 0], (np.float, np.float64)):
                return True
        else:
            if isinstance(img[0, 0, 0], (np.float, np.float64)):
                return True
    if isinstance(img, (tuple, list)):
        # a single color
        if isinstance(img[0], float):
            return True
    if isinstance(img, float):
        return True
    return False


def defaultLoader(f: typing.Union[str, typing.IO]) -> Image.Image:
    """
    load an image from a file-like object, filename, or url of type
        file://
        ftp://
        sftp://
        http://
        https://
    """
    if isinstance(f, (np.ndarray, Image.Image)):
        return f
    if f is None:
        return Image.new('RGBA', (1, 1), (255, 255, 255, 0))
    if isinstance(f, str):
        proto = f.split('://', 1)
        if len(proto) > 2 and proto.find('/') < 0:
            if proto[0] == 'file':
                f = proto[-1]
                if os.sep != '/':
                    f = f.replace('/', os.sep)
                    if f[1] == '|':
                        f[1] = ':'
            else:
                proto = proto[1]
        else:
            proto = 'file'
        if proto != 'file':
            import urllib.request
            import urllib.error
            import urllib.parse
            import io
            headers = {'User-Agent': 'Mozilla 5.10'}  # some servers only like "real browsers"
            request = urllib.request.Request(f, None, headers)
            response = urllib.request.urlopen(request)
            f = io.StringIO(response.read().decode('utf-8'))
    return Image.open(f)


def numpyArray(img, floatingPoint: bool = True, loader=None) -> np.ndarray:
    """
    always return a writeable ndarray

    if img is a pil image, convert it.
    if it's already an array, return it

    :param img: can be a pil image, a numpy array, or anything loader can load
    :param floatingPoint: return a float array vs return a byte array
    :param loader: return a tool used to load images from strings.  if None, use
        defaultLoader() in this file
    """
    if isinstance(img, str) or hasattr(img, 'read'):
        if loader is None:
            loader = defaultLoader
        img = loader(img)
    if isinstance(img, np.ndarray):
        a = img
    else:
        a = np.asarray(img)
    if not a.flags.writeable:
        a = a.copy()
    if a is not None and isFloat(a) != floatingPoint:
        if floatingPoint:
            a = a / 255.0
        else:
            a = np.int(a * 255)
    return a


def rgb2cmykArray(rgb):
    """
    Takes [[[r,g,b]]] colors in range 0..1
    Returns [[[c,m,y,k]]] in range 0..1
    """
    # k=rgb.sum(-1)
    c = 1.0 - rgb[:, :, 0]
    m = 1.0 - rgb[:, :, 1]
    y = 1.0 - rgb[:, :, 2]
    minCMY = np.dstack((c, m, y)).min(-1)
    c = (c - minCMY) / (1.0 - minCMY)
    m = (m - minCMY) / (1.0 - minCMY)
    y = (y - minCMY) / (1.0 - minCMY)
    k = minCMY
    a = 1 - rgb[:, :, 3]
    return np.dstack((c, m, y, k, a))


def clampImage(img, minimum=None, maximum=None):
    """
    Clamp an image's pixel to a valid color range

    :param img: clamp a numpy image to valid pixel values can be a PIL image for "do nothing"
    :param minimum: minimum value to clamp to (default is 0)
    :param maximum: maximum value to clamp to (default is the maximum pixel value)
    """
    if minimum is None:  # assign default
        if isFloat(img):
            minimum = 0.0
        else:
            minimum = 0
    elif isFloat(minimum) != isFloat(img):  # make sure it matches the image's number space
        if isFloat(minimum):
            minimum = int(minimum * 255)
        else:
            minimum = minimum / 255.0
    if maximum is None:  # assign default
        if isFloat(img):
            maximum = 1.0
        else:
            maximum = 255
    elif isFloat(maximum) != isFloat(img):  # make sure it matches the image's number space
        if isFloat(maximum):
            maximum = int(maximum * 255)
        else:
            maximum = maximum / 255.0
    if isinstance(img, np.ndarray):
        if minimum == 0 and (maximum >= 255 or (maximum >= 1.0 and isFloat(maximum))):
            # because conversion implies clamping to a valid range
            return img
        img = numpyArray(img)
    # print(img.shape,minimum,maximum)
    return np.clip(img, minimum, maximum)


def imageMode(img) -> str:
    """
    :param img: can either one be a textual image mode, numpy array, or an image

    :return bool:
    """
    if isinstance(img, str):
        return img
    if isinstance(img, np.ndarray):
        if len(img.shape) < 3:  # black and white is [x,y,val] not [x,y,[val]]
            return 'L'
        modeGuess = ['L', 'LA', 'RGB', 'RGBA', "CMYK"]
        return modeGuess[img.shape[2] - 1]
    return img.mode


def pilImage(img, loader=None) -> Image:
    """
    converts anything to a pil image

    :param img: can be a pil image, loadable file path, or a numpy array
    """
    if isinstance(img, Image.Image):
        # already what we need
        pass
    elif isinstance(img, str) or hasattr(img, 'read'):
        # load it with the loader
        if loader is None:
            loader = defaultLoader
        img = loader(img)
    else:
        # convert numpy array
        img = clampImage(img)
        mode = imageMode(img)
        if isFloat(img):
            img = np.round(img * 255)
        img = Image.fromarray(img.astype('uint8'), mode)
        # img = Image.fromarray(img.astype('uint8'), "RGBA")

    return img


def rgb2cmykImage(img):
    """
    same as rgb2cmykArray only takes an image and returns an image

    :param img: image to convert

    :return img: returns an "rgba" image, that is actually "cmyk"
    """
    rgb = numpyArray(img)
    final = rgb2cmykArray(rgb)
    return pilImage(final)


def png2pdf(root=r'.\\'):
    for img in sorted(glob.glob("{}/*.png".format(root))):
        img_name = img.split('\\')[-1].split('.')[0]
        doc = fitz.open()
        imgdoc = fitz.open(img)
        pdfbytes = imgdoc.convertToPDF()
        L
        imgpdf = fitz.open("pdf", pdfbytes)
        doc.insertPDF(imgpdf)
        doc.save(root + '\\' + img_name + '.pdf')
        doc.close()
        print('Convert ' + img_name + ' to ' + root + img_name + '.pdf')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="../data/jd/jd3_rgb.jpg")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # print(Image.open("tt.tiff").mode)
    rgb2cmykImage(args.path).save("tt.tiff")
