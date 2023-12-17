'''
@Time    : 2021/8/31 9:39
@Author  : leeguandon@gmail.com
'''
try:
    import io
    import pygame

    from pygame import freetype
    from pathlib import Path, PurePath
    from collections import defaultdict, OrderedDict
    from addict import Dict
    from skimage import color
    from PIL import Image, ImageDraw
    from urllib.request import urlopen

except ImportError:
    import warnings
    warnings.warn("export no support lib")
