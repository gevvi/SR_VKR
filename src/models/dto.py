from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ImageTask:
    input_path: Path
    output_path: Path
    scale: int


@dataclass
class ProcessingResult:
    input_path: str
    output_path: str
    success: bool
    message: str
    width_in: Optional[int] = None
    height_in: Optional[int] = None
    width_out: Optional[int] = None
    height_out: Optional[int] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    processing_time_sec: Optional[float] = None

    def to_dict(self):
        return asdict(self)
