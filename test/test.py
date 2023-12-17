'''
@Time    : 2022/7/8 11:24
@Author  : leeguandon@gmail.com
'''
import io
from psd_tools import PSDImage
from PIL import Image


def pilimage_tree(layer, pilimage, index):
    if layer.is_group():
        layer.name = "group-" + str(index)
        index += 1
        for sublayer in layer:
            pilimage_tree(sublayer, pilimage, index)
    else:
        pilimage.append(layer)
    return pilimage, index


def getpilimage(psd):
    pilimage = []
    index = 0
    for layer in psd:
        pilimage, index = pilimage_tree(layer, pilimage, index)

    return pilimage


# img = Image.open()
# psd = PSDImage.open(r'E:\common_tools\format_conversion\test\高分辨率图.psd')

# for layer in psd:
#     img = layer.topil()
#     output_buffer = io.BytesIO()
#     img.save(output_buffer, dpi=(300, 300), format='PNG')
#     if layer.name == 'background':
#         # img.save('gaoqing.png', quality=95, format='PNG')
#         img.save('gaoqingtu.png', dpi=(300, 300))
import json

def get_json(json_path):
    with open(json_path, encoding='utf-8') as f:
        input_json = json.load(f)

    print(input_json.keys())
    # if "requestBody" in input_json.keys():
    #     return {
    #         "psdFiles": input_json["requestBody"]["psdFiles"],
    #         "objectStoreInfo": input_json["requestBody"]["objectStoreInfo"],
    #     }
    return input_json

input_json = get_json("local_dpa.json")
material = input_json['requestBody']['materialList']['layers']['elements']

for mate in material:
    if 'text' not in mate:
        url1 = material[mate]['materialUrl']
        pass








