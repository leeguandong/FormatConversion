'''
@Time    : 2022/1/6 10:44
@Author  : leeguandon@gmail.com
'''
import os
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from fcn import build_rgb2cmyk, NotSupportFormatError


def parse_args():
    parser = argparse.ArgumentParser(description="Format transform")
    parser.add_argument("--format", default="PILICCRGBA2CMYK", choices=["PILICCRGBA2CMYK"], type=str)
    parser.add_argument("--indir", default="../data/batch_jd", type=str)
    parser.add_argument("--saved", default="../output1", type=str)
    args = parser.parse_args([])
    return args


def main():
    args = parse_args()

    indir = args.indir
    saved = args.saved
    Path(saved).mkdir(parents=True, exist_ok=True)

    file = Path(indir).rglob("*.png")

    start = time.time()
    for index, file_png in tqdm(enumerate(file)):
        img = build_rgb2cmyk(dict(type=args.format))(file_png)
        if img.mode == "RGB":
            img.save(str(Path(saved, str(index) + "_" + file_png.stem + "_pil_icc.jpg")), dpi=(300, 300))
        elif img.mode == "CMYK":
            img.save(str(Path(saved, str(index) + "_" + file_png.stem + "_pil_icc.tiff")), dpi=(300, 300))
            # img.save(str(Path(saved, file_png.stem + "_pil_icc.tiff")), qulity=100)
        else:
            raise NotSupportFormatError
    print('total time', round(time.time() - start, 2))
    print("avg time", round((time.time() - start) / len(os.listdir(indir))))


if __name__ == "__main__":
    main()
