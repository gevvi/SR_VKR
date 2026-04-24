import csv
from pathlib import Path
from typing import Iterable

from models.dto import ProcessingResult


class ResultSaver:
    def save_image(self, image, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    def save_report(self, results: Iterable[ProcessingResult], report_path: Path):
        rows = [r.to_dict() for r in results]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
