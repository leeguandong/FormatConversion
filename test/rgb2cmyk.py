'''
@Time    : 2022/1/6 10:10
@Author  : leeguandon@gmail.com
'''

import torch
import argparse
import cv2
import pylab
# import PythonMagick
import numpy as np
import operator, itertools
from PIL import Image, ImageCms
from scipy import misc
# from wand.image import Image
from wand.color import Color
# from libtiff import TIFF
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

torch.manual_seed(2022)


def png2pdf(path):
    png = Image.open(path)
    png.load()  # required for png.split()

    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    # bg.save("../data/jd")
    img = background.convert("CMYK")
    img.save("../data/jd/jd3_pil_rgba2rgb_pdf.pdf")


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def pil_rgba_rgb(path):
    png = Image.open(path)
    png.load()  # required for png.split()

    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    # bg.save("../data/jd")
    img = background.convert("CMYK")
    img.save("../data/jd/jd3_pil_rgba2rgb_cmyk.tiff")

    # png = Image.open(path).convert('RGBA')
    # background = Image.new('RGBA', png.size, (255, 255, 255))
    # alpha_composite = Image.alpha_composite(background, png)
    # alpha_composite.save('foo.jpg', 'JPEG', quality=80)


def pil_rgba_cmyk(path):
    png = Image.open(path)
    png.load()

    img = png.convert("CMYK")
    background = Image.new("CMYK", png.size, (255, 255, 255))
    background.paste(img, mask=png.split()[3])

    img.save("../data/jd/jd3_pil_paste_cmyka.tiff")


def pil2rgb(path):
    img = Image.open(path)
    print(img.mode)
    # img.show()
    # img = img.convert("CMYK")
    # img.save("../data/jd/jd3_rgba.tiff")
    img = img.convert("RGB")
    img.save("../data/jd/jd3_rgb.jpg")


def alpha_rgba(path):
    """
    可以通过将图像转换为字符串(例如，从图像中获取alpha数据并将其保存为灰度图像)，
    从而一次从整个图像中获取alpha数据。)
    :param path:
    :return:
    """
    img = Image.open(path)
    r, g, b, a = img.split()
    print(a)
    rgbData = img.tobytes("raw", "RGB")
    print(len(rgbData))
    alphaData = img.tobytes("raw", "A")
    print(len(alphaData))
    alphaImage = Image.frombytes("L", img.size, alphaData)
    alphaImage.save("../data/jd/jd3_alpha.png")


def get_alpha_channel(path):
    "Return the alpha channel as a sequence of values"
    image = Image.open(path)
    # first, which band is the alpha channel?
    try:
        alpha_index = image.getbands().index('A')
    except ValueError:
        image = image.convert('RGBA')
        alpha_index = image.getbands().index('A')

    alpha_getter = operator.itemgetter(alpha_index)
    img = map(alpha_getter, image.getdata())
    # img.save("../data/jd/jd_alpha1.png")
    print(img)


def pil_rgba2tiff(path):
    img = Image.open(path)
    print(img.mode)
    img = img.convert("CMYK")
    print(img.mode)


def pilrgba2rga2tiff(path):
    img = Image.open(path)
    print(img.mode)
    img = img.convert("RGB")
    print(img.mode)
    img = img.convert("CMYK")
    print(img.mode)
    img.save("../data/jd/jd_pil_rgba_rgb_cmyk.tiff")


def pilrgb2tiff(path):
    img = Image.open(path)
    print(img.mode)
    img = img.convert("CMYK")
    print(img.mode)
    print(img.size)
    img.save("../data/jd/jd3_pil_rgb_cmyk.tiff")

    # cv_img = cv2.imread(path)
    # Y, M, C, K = cv2.split(cv_img)
    # alpha = img.split()[-1]
    # Image.merge()
    # pass


def pil2tiffcv(path):
    img = Image.open(path)
    print(img.mode)
    img = img.convert("CMYK")
    print(img.mode)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # cv2.imshow('CMYK', opencvImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("../data/jd/jd3_cv.tiff", opencvImage, ((int(cv2.IMWRITE_TIFF_RESUNIT), 2,
    #                                                           int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
    #                                                           int(cv2.IMWRITE_TIFF_XDPI), 100,
    #                                                           int(cv2.IMWRITE_TIFF_YDPI), 100)))
    cv2.imwrite("../data/jd/jd3_cv.tiff", opencvImage)


def readcmyk(path):
    """
    支持单通道及多通道Uint8 TIFF图像读取，读取单通道Uint16 TIFF图像转为Uint8处理，
    直接读取Uint16 TIFF多通道图像出错
    :param path:
    :return:
    """
    img = cv2.imread(path, 2)
    # plt.figure(dpi=180)  # 显示图像
    # plt.imshow(img)
    # pylab.show()
    cv2.imshow('image', img)
    cv2.waitKey(0)  # 0讲无限期等待键盘输入，也用来检测是否被按下
    cv2.destroyAllWindows()  # 删除所有建立的窗口
    # cv2.imwrite(img,"../data/jd/jd3_cv_rgba_cmyk.tiff")


def rgba2cmyk(path):
    image = Image.open(path)
    np_image = np.array(image)
    copy = np_image.copy()
    copy[:, :, 0], copy[:, :, 2] = np_image[:, :, 2], np_image[:, :, 0]
    img = Image.fromarray(copy).convert('CMYK')
    img.save("../data/jd/jd3_pil_rgba_cmyk.tiff")


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
        modeGuess = ['L', 'LA', 'RGB', 'RGBA']
        return modeGuess[img.shape[2] - 1]
    return img.mode


def hasAlpha(mode) -> bool:
    """
    determine if an image mode is has an alpha channel

    :param mode: can either one be a textual image mode, numpy array, or an image
    :return bool:
    """
    if not isinstance(mode, str):
        mode = imageMode(mode)
    return mode[-1] == 'A'


def getAlpha(image, alwaysCreate: bool = True):
    """
    gets the alpha channel regardless of image type

    :param image: the image whose mask to get
    :param alwaysCreate: always returns a numpy array (otherwise, may return None)

    :return: alpha channel as a PIL image, or numpy array,
        or possibly None, depending on alwaysCreate
    """
    ret = None
    if image is None or not hasAlpha(image):
        if alwaysCreate:
            ret = np.array(image.size())
            ret.fill(1.0)
    elif isinstance(image, Image.Image):
        ret = image.getalpha()
    else:
        ret = image[:, :, -1]
    return ret


def cmyka(path):
    img = Image.open(path)
    print(img.mode)
    img_cmyk = img.convert("CMYK")
    print(img_cmyk.mode)

    alpha = getAlpha(img)
    pass


def rgb_cmy(path):
    img = cv2.imread(path)
    r, g, b = cv2.split(img)  # split the channels
    # black = min(1 - r, 1 - g, 1 - b)
    # normalization [0,1]
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    c = 1 - r
    m = 1 - g
    y = 1 - b

    # img_CMY = cv2.merge((c, m, y, k))  # merge the channels
    # img_NEW = img_CMY * 255
    img = (np.dstack((c, m, y)) * 255).astype(np.uint8)
    # cv2.imshow('CMY', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("../data/jd/jd3_rgb_cmy.tiff", img)


def rgb_cmyk(path):
    """ `img` must have (m,n,3) dimensions
    R' = R/255
    G' = G/255
    B' = B/255
    The black key (K) color is calculated from the red (R'), green (G') and
    blue (B') colors:
    K = min(R', G', B')
    The cyan color (C) is calculated from the red (R') and black (K) colors:
    C = (1-R'-K) / (1-K)
    The magenta color (M) is calculated from the green (G') and black (K)
    colors:
    M = (1-G'-K) / (1-K)
    The yellow color (Y) is calculated from the blue (B') and black (K) colors:
    Y = (1-B'-K) / (1-K)
    """
    img = Image.open(path)
    img = np.asarray(img)
    r = img[..., 0] / 255.
    g = img[..., 1] / 255.
    b = img[..., 2] / 255.
    #
    k = 1 - np.max((r, g, b), axis=0)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    #
    cmyk = np.dstack((c, m, y, k)) * 255
    img = cmyk.astype('uint8')

    cv2.imwrite("../data/jd/jd3_rgb_cmyk_cv_2.tiff", img)


def cv_rgb_cmyk(path):
    bgr = cv2.imread(path)  # your bgr image
    bgrdash = bgr.astype(np.float) / 255.

    K = 1 - np.max(bgrdash, axis=2)
    C = (1 - bgrdash[..., 2] - K) / (1 - K)
    M = (1 - bgrdash[..., 1] - K) / (1 - K)
    Y = (1 - bgrdash[..., 0] - K) / (1 - K)

    img = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)

    cv2.imshow('CMYK', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("../data/jd/jd3_rgb_cmyk_cv.tiff", img)


def cv_rgb_cmyk_1(path):
    # Import image
    img = plt.imread(path)

    # Create float
    bgr = img.astype(float) / 255.

    # Extract channels
    with np.errstate(invalid='ignore', divide='ignore'):
        K = 1 - np.max(bgr, axis=2)
        C = (1 - bgr[..., 2] - K) / (1 - K)
        M = (1 - bgr[..., 1] - K) / (1 - K)
        Y = (1 - bgr[..., 0] - K) / (1 - K)

    # Convert the input BGR image to CMYK colorspace
    CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)

    # Split CMYK channels
    Y, M, C, K = cv2.split(CMYK)

    np.isfinite(C).all()
    np.isfinite(M).all()
    np.isfinite(K).all()
    np.isfinite(Y).all()

    # Save channels
    cv2.imwrite('../data/jd/jd3_rgb_cmyk_cv_C.jpg', C)
    cv2.imwrite('../data/jd/jd3_rgb_cmyk_cv_M.jpg', M)
    cv2.imwrite('../data/jd/jd3_rgb_cmyk_cv_Y.jpg', Y)
    cv2.imwrite('../data/jd/jd3_rgb_cmyk_cv_K.jpg', K)
    # cv2.imshow('CMYK', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("../data/jd/jd3_rgb_cmyk_cv_1.tiff", CMYK)


def readtiff(path):
    input = TIFF.open(path).read_image()

    def reduceDepth(image, display_min, display_max):
        image -= display_min
        image = np.floor_divide(image, (display_min - display_max + 1) / 256)
        image = image.astype(np.uint8)
        return image

    v8 = reduceDepth(input, 0, 65536)

    im = Image.fromarray(v8)
    im = im.convert('RGB')
    im.save("cmyk-out.jpeg")


def mping_rgb2cmyk(path, percent_gray=100):
    rgb = mpimg.imread(path, 0)
    cmy = 1 - rgb / 255.0
    k = np.min(cmy, axis=2) * (percent_gray / 100.0)
    k[np.where(np.sum(rgb, axis=2) == 0)] = 1.0  # anywhere there is no color, set the k chanel to max
    k_mat = np.stack([k, k, k], axis=2)

    with np.errstate(divide='ignore', invalid='ignore'):
        cmy = (cmy - k_mat) / (1.0 - k_mat)
        cmy[~np.isfinite(cmy)] = 0.0
    img = np.dstack((cmy, k))
    # plt.imshow(img)
    plt.savefig('../data/jd3_mping_rgba_cmyk.tiff')
    # plt.show()


def cmyk2rgb(path):
    image = Image.open(path)
    # sRGB Color Space Profile.icm windows自带的；
    if image.mode == 'CMYK':
        image = ImageCms.profileToProfile(image, 'USWebCoatedSWOP.icc', 'sRGB Color Space Profile.icm', renderingIntent=0,
                                          outputMode='RGB')
    image.save("../data/jd/jd3_cmyk_rgb.jpg")


def pil_cms_cmyk2rgb(path):
    image = Image.open(path)
    if image.mode == 'RGB':
        image = ImageCms.profileToProfile(image, 'sRGB Color Space Profile.icm', 'USWebCoatedSWOP.icc', renderingIntent=0,
                                          outputMode='CMYK')
    image.save("../data/jd/jd3_cms_rgb_cmyk.tiff")


def pil_cmyk2rgb(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img.save("../data/jd/jd3_pil_cmyk_rgb.jpg")


def wand_rgb_cmyk(path):
    with Image(filename=path) as img:
        # print('width =', img.width)
        # print('height =', img.height)
        print("format =", img.format)
        # img.format = "tiff"
        # img.save(filename="../data/jd/jd3_wand_rgb_tiff.tiff")
        img.transform_colorspace("cmyk")
        img.save(filename="../data/jd/jd3_wand_rgb_cmyk.tiff")


def wand_rgb_cmyk2(path):
    with Image(filename=path) as img:
        img.type = 'truecolor'
        img.alpha_channel = True
        img.colorspace = 'cmyk'
        img.save(filename='../data/jd/jd3_wand_rgb_cmyk1.tiff')


def wand_rgba_cmky3(path):
    with Image(filename=path, background=Color("transparent")) as img:
        # img.format = "png"
        # img.alpha_channel = True
        img.colorspace = 'cmyk'
        img.save(filename='../data/jd/jd3_wand_rgb_cmyk2.tiff')


def pythonmagick_rgba_cmyk(path):
    def get_color_blob(file_name):
        color = PythonMagick.Blob()
        im = PythonMagick.Image(file_name)
        im.write(color, "icc")
        return color

    cmyk = get_color_blob(r"E:\common_tools\format_conversion\config\USWebCoatedSWOP.icc")
    temp = PythonMagick.Image(path)
    temp.iccColorProfile(cmyk)
    temp.write("../data/jd/jd3_pythonmagic_rgb_cmyk.tiff")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="../data/batch_jd/1 原图46bf83c59f9e0181.png")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # pil2rgb(args.path)
    # alpha_rgba(args.path)
    # get_alpha_channel(args.path)
    # readcmyk(args.path)
    # pilrgba2rga2tiff(args.path)
    # pilrgb2tiff(args.path)
    # pil_rgba_rgb(args.path)
    # pil_rgba_cmyk(args.path)
    # readcmyk(args.path)
    # mping_rgb2cmyk(args.path)
    # cmyk2rgb(args.path)
    # pil_cmyk2rgb(args.path)
    # rgb_cmy(args.path)
    # rgb_cmyk(args.path)
    # pil2tiffcv(args.path)
    # cv_rgb_cmyk(args.path)
    # cv_rgb_cmyk_1(args.path)
    # pilrgb2tiff(args.path)
    # rgb_cmyk(args.path)
    # rgba2cmyk(args.path)
    # readtiff(args.path)
    # cmyka(args.path)
    # pil_cms_cmyk2rgb(args.path)
    # wand_rgb_cmyk(args.path)
    # wand_rgb_cmyk2(args.path)
    # wand_rgba_cmky3(args.path)
    # pythonmagick_rgba_cmyk(args.path)

    png2pdf(args.path)

    # print(Image.open(r"F:\Dataset\qiantu\Aus_03.tif").mode)
