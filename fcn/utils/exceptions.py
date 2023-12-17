'''
@Time    : 2022/1/9 16:26
@Author  : leeguandon@gmail.com
'''


class Error(Exception):
    def __init__(self, *args, **kwargs):
        super(Error, self).__init__(*args, **kwargs)


class ShapeError(Error):
    pass


class NotSupportFormatError(Error):
    def __init__(self):
        print("not support this format")
