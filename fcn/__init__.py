'''
@Time    : 2022/1/6 10:04
@Author  : leeguandon@gmail.com
'''

from .builder import RGB2CMYK, build_rgb2cmyk
from .utils import NotSupportFormatError
from .core import PILICCRGBA2CMYK

__all__ = [
    "build_rgb2cmyk", "RGB2CMYK",
    "NotSupportFormatError",
    "PILICCRGBA2CMYK"
]

"""
1.几种主流的图片格式转换方法
rgb2cmyk


rgb2rgba


rgba2rgb





"""
