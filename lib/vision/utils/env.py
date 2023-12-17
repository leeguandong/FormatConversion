'''
@Time    : 2021/8/31 19:25
@Author  : leeguandon@gmail.com
'''
import sys
import cv2
import torch
import subprocess
import os.path as osp

from lib import vision
from collections import defaultdict


def collect_env():
    env_info = {}
    env_info["sys.platform"] = sys.platform
    env_info["Python"] = sys.version.replace("\n", "")

    cuda_available = torch.cuda.is_available()
    env_info["CUDA available"] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        from lib.mmcv import _get_cuda_home
        CUDA_HOME = _get_cuda_home()
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    # try:
    #     gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    #     gcc = gcc.decode('utf-8').strip()
    #
    #     env_info['GCC'] = gcc
    # except subprocess.CalledProcessError:  # gcc is unavailable
    #     env_info['GCC'] = 'n/a'

    if "win" in sys.platform:
        env_info['GCC'] = 'n/a'
    else:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()

        env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    # env_info['PyTorch compiling details'] = get_build_config()

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info['OpenCV'] = cv2.__version__

    env_info['MMCV'] = vision.__version__

    return env_info
