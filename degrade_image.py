import os
import cv2
import math
import random
import numpy as np
import torch
from torch.nn import functional as F
from basicsr.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    circular_lowpass_kernel,
)
from basicsr.utils.img_util import img2tensor, tensor2img

def gaussian_kernel2d(kernel_size, sigma_x, sigma_y, rotation=0.0):
    """Создаёт 2D-гауссово ядро с анизотропными сигмами и поворотом."""
    k = kernel_size
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    # Поворот координат
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    x_rot = cos_r * xx + sin_r * yy
    y_rot = -sin_r * xx + cos_r * yy

    kernel = np.exp(-0.5 * ((x_rot ** 2) / (sigma_x ** 2) + (y_rot ** 2) / (sigma_y ** 2)))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

# Параметры деградации (упрощённый вариант из статьи Real-ESRGAN) [web:61][web:133]
opt = {
    'scale': 4,

    # 1‑я ступень – почти без изменения геометрии и блюра
    'resize_prob': [0.0, 0.2, 0.8],      # чаще keep, иногда лёгкий down
    'resize_range': [0.9, 1.1],          # масштаб практически 1
    'gaussian_noise_prob': 0.3,
    'noise_range': [0.3, 1.5],           # очень слабый шум
    'poisson_scale_range': [0.0, 0.3],   # почти нет пуассоновского
    'gray_noise_prob': 0.0,              # только цветной шум
    'jpeg_range': [80, 100],             # лёгкое JPEG-сжатие

    # 2‑ю ступень фактически выключаем
    'second_blur_prob': 0.0,
    'resize_prob2': [0.0, 0.0, 1.0],     # всегда keep
    'resize_range2': [1.0, 1.0],
    'gaussian_noise_prob2': 0.0,
    'noise_range2': [0.0, 0.0],
    'poisson_scale_range2': [0.0, 0.0],
    'gray_noise_prob2': 0.0,
    'jpeg_range2': [80, 100],
}
def generate_kernels():
    """Генерация трёх ядер (blur1, blur2, sinc) без вызовов random_mixed_kernels."""
    kernel_range = [7, 9, 11, 13, 15, 17, 19, 21]

    # ------------------------ First degradation kernel ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < 0.1:  # небольшая вероятность взять sinc
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        sigma = np.random.uniform(0.2, 3.0)
        if np.random.rand() < 0.5:
            # изотропная гауссиана
            kernel = gaussian_kernel2d(kernel_size, sigma, sigma, rotation=0.0)
        else:
            # анизотропная с поворотом
            sigma_x = np.random.uniform(0.2, 3.0)
            sigma_y = np.random.uniform(0.2, 3.0)
            rot = np.random.uniform(-math.pi, math.pi)
            kernel = gaussian_kernel2d(kernel_size, sigma_x, sigma_y, rotation=rot)

    pad_size = (21 - kernel_size) // 2
    kernel1 = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Second degradation kernel ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < 0.1:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        sigma = np.random.uniform(0.2, 1.5)
        if np.random.rand() < 0.5:
            kernel = gaussian_kernel2d(kernel_size, sigma, sigma, rotation=0.0)
        else:
            sigma_x = np.random.uniform(0.2, 1.5)
            sigma_y = np.random.uniform(0.2, 1.5)
            rot = np.random.uniform(-math.pi, math.pi)
            kernel = gaussian_kernel2d(kernel_size, sigma_x, sigma_y, rotation=rot)

    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Final sinc kernel ------------------------ #
    if np.random.uniform() < 0.8:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
    else:
        sinc_kernel = np.zeros((21, 21), dtype=np.float32)
        sinc_kernel[10, 10] = 1.0

    kernel1 = torch.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
    kernel2 = torch.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
    sinc_kernel = torch.FloatTensor(sinc_kernel).unsqueeze(0).unsqueeze(0)

    return kernel1, kernel2, sinc_kernel

def filter2D(img, kernel):
    # img: (B, C, H, W), kernel: (1, 1, k, k)
    B, C, H, W = img.shape
    k = kernel.shape[-1]
    pad = k // 2
    img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    # применяем одно и то же ядро ко всем каналам
    kernel = kernel.to(img.device)
    kernel = kernel.expand(C, 1, k, k)
    out = F.conv2d(img_pad, kernel, groups=C)
    return out


def degrade_tensor(gt, opt):
    """Повторяет двухступенчатую деградацию Real-ESRGAN для одного GT-тензора (1,C,H,W). [web:134]"""
    device = gt.device
    kernel1, kernel2, sinc_kernel = generate_kernels()
    kernel1 = kernel1.to(device)
    kernel2 = kernel2.to(device)
    sinc_kernel = sinc_kernel.to(device)

    ori_h, ori_w = gt.shape[2:4]

    # 1-я ступень: blur -> resize -> noise -> JPEG [web:61][web:133]
    out = filter2D(gt, kernel1)

    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)

    gray_noise_prob = opt['gray_noise_prob']
    if np.random.uniform() < opt['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=opt['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False
        )

    # здесь можно вставить DiffJPEG, если очень нужно полностью повторить оригинал;
    # для минимального примера ограничимся clamp [web:134]
    out = torch.clamp(out, 0, 1)

    # 2-я ступень: опциональный blur -> resize -> noise -> JPEG + sinc [web:61][web:133]
    if np.random.uniform() < opt['second_blur_prob']:
        out = filter2D(out, kernel2)

    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range2'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out,
        size=(
            int(ori_h / opt['scale'] * scale),
            int(ori_w / opt['scale'] * scale)
        ),
        mode=mode
    )

    gray_noise_prob = opt['gray_noise_prob2']
    if np.random.uniform() < opt['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=opt['noise_range2'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False
        )

    # Финальный resize до (ori/scale) и sinc-фильтр [web:134]
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out,
        size=(ori_h // opt['scale'], ori_w // opt['scale']),
        mode=mode
    )
    out = filter2D(out, sinc_kernel)

    # имитация 8-битного квантизатора [web:133]
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    return lq


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='GT image path')
    parser.add_argument('-o', '--output', type=str, default='degraded.png', help='Output LQ image path')
    args = parser.parse_args()

    # читаем GT [web:41]
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {args.input}')

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    img = img.astype(np.float32) / 255.0
    img_t = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).cuda()

    with torch.no_grad():
        lq_t = degrade_tensor(img_t, opt)

    lq = tensor2img(lq_t, rgb2bgr=True, out_type=np.uint8)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, lq)
    print(f'Saved degraded image to {args.output}')


if __name__ == '__main__':
    main()