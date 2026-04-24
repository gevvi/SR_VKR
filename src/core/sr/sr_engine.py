import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


class SuperResolutionEngine:
    def __init__(self):
        self.model_repo_path: Path | None = None
        self.model_name: str | None = None
        self.ready = False
        self.upsampler = None
        self.device = None
        self.outscale = 4

    def configure(self, model_repo_path: Path | None, model_name: str):
        self.model_repo_path = model_repo_path
        self.model_name = model_name
        self.upsampler = None
        self.ready = False
        self.outscale = 4

        if model_repo_path is None or not model_repo_path.exists():
            return

        repo_path = Path(model_repo_path).resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = (model_name or 'RealESRGAN_x4plus').split('.')[0]
        self.outscale = 4

        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_name = 'RealESRGAN_x4plus.pth'
        elif model_name == 'RealESRNet_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_name = 'RealESRNet_x4plus.pth'
        elif model_name == 'RealESRGAN_x4plus_anime_6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_name = 'RealESRGAN_x4plus_anime_6B.pth'
        elif model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_name = 'RealESRGAN_x2plus.pth'
            self.outscale = 2
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_name = 'RealESRGAN_x4plus.pth'
            self.outscale = 4

        candidate_paths = [
            repo_path / 'weights' / file_name,
            repo_path / 'experiments' / 'pretrained_models' / file_name,
            repo_path / file_name,
        ]
        model_path = next((str(p) for p in candidate_paths if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                f'Не найден файл весов {file_name}. Ожидались пути: ' + ', '.join(str(p) for p in candidate_paths)
            )

        half = self.device.type == 'cuda'
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
            gpu_id=0 if self.device.type == 'cuda' else None,
        )
        self.ready = True

    def upscale(self, image: Image.Image, scale: int) -> Image.Image:
        if not self.ready or self.upsampler is None:
            return image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)

        img = np.array(image)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = img[:, :, ::-1]
        output, _ = self.upsampler.enhance(img, outscale=scale)
        output = output[:, :, ::-1]
        return Image.fromarray(output)
