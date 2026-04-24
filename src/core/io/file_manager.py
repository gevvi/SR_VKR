from pathlib import Path
from typing import Iterable, List

from config.settings import ProcessingSettings
from models.dto import ImageTask


class FileManager:
    def list_images(self, input_dir: Path, allowed_extensions: Iterable[str]) -> List[Path]:
        exts = {ext.lower() for ext in allowed_extensions}
        return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    def build_tasks(self, settings: ProcessingSettings) -> List[ImageTask]:
        if settings.input_dir is None or settings.output_dir is None:
            raise ValueError('Input and output directories must be selected')

        settings.output_dir.mkdir(parents=True, exist_ok=True)
        files = self.list_images(settings.input_dir, settings.allowed_extensions)
        tasks: List[ImageTask] = []
        for src in files:
            dst = settings.output_dir / src.name
            tasks.append(ImageTask(input_path=src, output_path=dst, scale=settings.scale))
        return tasks
