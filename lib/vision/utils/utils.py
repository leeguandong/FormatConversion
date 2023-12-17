'''
@Time    : 2021/8/31 9:43
@Author  : leeguandon@gmail.com
'''
import numpy as np


def box_shift(bbox, shift_x, shift_y):
    x1, y1, x2, y2 = bbox
    x1 += shift_x
    y1 += shift_y
    x2 += shift_x
    y2 += shift_y
    return [x1, y1, x2, y2]


def get_image_truebbox(image):
    """
    #得到透明图的商品区域
    :param image: PIL
    :return: x1, y1, x2, y2, w, h: 透明图中商品的左上角坐标x1, y1、右下角坐标x2, y2、宽度和高度w, h
    """
    if image.mode != 'RGBA':
        image = image.convert("RGBA")
    img_np = np.asarray(image)
    y, x = (img_np[:, :, 3] > 0).nonzero()
    x1, y1, x2, y2 = np.min(x), np.min(y), min(np.max(x) + 1, img_np.shape[1]), min(np.max(y) + 1, img_np.shape[0])
    w = x2 - x1
    h = y2 - y1
    return x1, y1, x2, y2, w, h


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_box(box):
    # box 中可能存在空 list
    box = [x for x in box if x != []]
    box = np.array(box)
    if len(box.shape) == 2:
        box = [int(max(box[:, 0].min(), 0)), int(max(box[:, 1].min(), 0)), int(box[:, 2].max()), int(box[:, 3].max())]
    else:
        raise ValueError(f"box shape error {box.shape}")
    return box


def paste2center(box, image):
    """
    将 image 不变形的放入 box 中，返回此时坐标
    :param box:
    :param image:
    :return:
    """
    if isinstance(image, (list, tuple)):
        w, h = image[-2], image[-1]
    else:
        if image.mode == "RGBA":
            image_np = np.array(image)
            y_in, x_in = (image_np[:, :, 3] > 0).nonzero()
            x = np.min(x_in)
            y = np.min(y_in)
            w = min(np.max(x_in) + 1, image_np.shape[1]) - x
            h = max(np.max(y_in) + 1, image_np.shape[0]) - y
        else:
            w, h = image.size

    bx1, by1, bx2, by2 = box[:4]
    bw = bx2 - bx1
    bh = by2 - by1
    # 一定要注意 bw/bh 是已存在的矩形框，w/h 是输入的图片，bw/w 则是矩形框比图片的最小值，其实是取矩形框内最小的一边进行 resize，
    # 能把图片放到矩形框最小一边为基准
    f = min(bw / w, bh / h)
    w1 = int(math.ceil(f * w))
    h1 = int(math.ceil(f * h))
    x1_out = int(math.ceil((bx1 + bx2) / 2 - w1 / 2))
    y1_out = int(math.ceil((by1 + by2) / 2 - h1 / 2))
    x2_out = x1_out + w1
    y2_out = y1_out + h1
    out_box = [x1_out, y1_out, x2_out, y2_out]
    return out_box


def calculate_min_rect(material_list):
    """
    计算 list 列表中的 最小外接矩形，和上面 get_box 大概是一个意思，不过此处的 material_list 覆盖 list 中是 Material 对象
    :param material_list:
    :return:
    """
    assert len(material_list) > 0, "no material is appointed"
    if isinstance(material_list[0], list):
        x1, y1, x2, y2 = material_list[0]
    else:
        x1, y1, x2, y2 = material_list[0].bbox[0], material_list[0].bbox[1], material_list[0].bbox[2], material_list[0].bbox[3]

    for item in material_list:
        if isinstance(item, list):
            x1 = min(x1, item[0])
            y1 = min(y1, item[1])
            x2 = max(x2, item[2])
            y2 = max(y2, item[3])
        else:
            x1 = min(x1, item.bbox[0])
            y1 = min(y1, item.bbox[1])
            x2 = max(x2, item.bbox[2])
            y2 = max(y2, item.bbox[3])
    min_rect = [x1, y1, x2, y2]
    return min_rect


def get_center_wh(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    return center_x, center_y, w, h


def start_align_hv(start, end, length, align):
    """
    按照对齐方式计算左上角坐标
    :param start:  start in x or y orientation
    :param end:  end in x or y orientation
    :param length:  text length also w in x or y orientation
    :param align:  [left,mid,right] or [top,mid,bottom]
    :return:
    """
    align = align.lower()
    if "left" in align or "top" in align:
        start = 0
    elif "right" in align or "bottom" in align:
        start = (end - start) - length
    elif "mid" in align:
        start = 1 / 2 * (end - start - length)
    else:
        raise TypeError("No matching align type")
    return start
