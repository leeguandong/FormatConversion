'''
@Time    : 2022/1/6 10:28
@Author  : leeguandon@gmail.com
'''
from config.config import *
from PIL import Image, ImageCms
from fcn import RGB2CMYK, NotSupportFormatError


@RGB2CMYK.register_module()
class PILICCRGBA2CMYK(object):
    sRGB = gflags.FLAGS.sRGB
    SWOP = gflags.FLAGS.SWOP

    def __init__(self, *args, **kwargs):
        super(PILICCRGBA2CMYK, self).__init__()

    def icc_rgb_cmyk(self, img):
        img = ImageCms.profileToProfile(img, self.sRGB, self.SWOP, renderingIntent=0, outputMode='CMYK')
        return img

    def icc_cmyk_rgb(self, img):
        img = ImageCms.profileToProfile(img, self.SWOP, self.sRGB, renderingIntent=0, outputMode='RGB')
        return img

    def __call__(self, img_path):
        img = Image.open(img_path)
        if img.mode == "RGB":
            img = self.icc_rgb_cmyk(img)
        elif img.mode == "RGBA":
            img.load()  # required for png.split()
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = self.icc_rgb_cmyk(background)
        elif img.mode == "CMYK":
            img = self.icc_cmyk_rgb(img)
        else:
            raise NotSupportFormatError

        return img
