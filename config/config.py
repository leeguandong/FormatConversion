'''
@Time    : 2022/1/9 16:05
@Author  : leeguandon@gmail.com
'''
import gflags

gflags.DEFINE_string("sRGB", "../config/sRGB Color Space Profile.icm", "")
gflags.DEFINE_string("SWOP", "../config/USWebCoatedSWOP.icc", "")
# gflags.DEFINE_string("sRGB", "../config/sRGB Color Space Profile.icm", "")
# gflags.DEFINE_string("SWOP", "../config/SWOP (Coated).icc", "")

gflags.FLAGS("")
