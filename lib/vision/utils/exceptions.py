'''
@Time    : 2021/8/31 9:39
@Author  : leeguandon@gmail.com
'''


class Error(Exception):
    def __init__(self, *args, **kwargs):
        super(Error, self).__init__(*args, **kwargs)


class ShapeError(Error):
    pass


class PathError(Error):
    pass
