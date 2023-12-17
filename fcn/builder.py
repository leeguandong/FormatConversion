'''
@Time    : 2022/1/6 10:27
@Author  : leeguandon@gmail.com
'''
from lib.vision.utils import Registry, build

RGB2CMYK = Registry("rgb2cmyk")


def build_rgb2cmyk(cfg, default_args=None):
    return build(cfg, RGB2CMYK, default_args)
