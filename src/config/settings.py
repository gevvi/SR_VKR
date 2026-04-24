from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

SUPPORTED_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


@dataclass
class ProcessingSettings:
    input_dir: Path | None = None
    output_dir: Path | None = None
    scale: int = 4
    save_metrics: bool = True
    metrics_enabled: bool = True
    normalize: bool = True
    convert_to_rgb: bool = True
    overwrite: bool = True
    model_name: str = 'RealESRGAN_x4plus'
    model_repo_path: Path | None = None
    report_name: str = 'processing_report.csv'
    allowed_extensions: Tuple[str, ...] = field(default_factory=lambda: SUPPORTED_EXTENSIONS)
