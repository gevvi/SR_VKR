import math
import numpy as np
from PIL import Image


class MetricsCalculator:
    def _to_np(self, image: Image.Image) -> np.ndarray:
        return np.asarray(image).astype(np.float32)

    def psnr(self, original: Image.Image, processed: Image.Image) -> float:
        a = self._to_np(original)
        b = self._to_np(processed.resize(original.size))
        mse = np.mean((a - b) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def ssim(self, original: Image.Image, processed: Image.Image) -> float:
        x = self._to_np(original.resize(processed.size)).mean(axis=2)
        y = self._to_np(processed).mean(axis=2)
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        mu_x, mu_y = x.mean(), y.mean()
        sigma_x, sigma_y = x.var(), y.var()
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
        num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        return float(num / den) if den else 1.0

    def lpips_placeholder(self, original: Image.Image, processed: Image.Image) -> float:
        a = self._to_np(original.resize(processed.size)) / 255.0
        b = self._to_np(processed) / 255.0
        return float(np.mean(np.abs(a - b)))
