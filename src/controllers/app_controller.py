import time
from pathlib import Path
from typing import Callable, List, Optional

from config.settings import ProcessingSettings
from core.io.file_manager import FileManager
from core.metrics.metrics_calculator import MetricsCalculator
from core.preprocess.image_preprocessor import ImagePreprocessor
from core.save.result_saver import ResultSaver
from core.sr.sr_engine import SuperResolutionEngine
from models.dto import ProcessingResult
from utils.logger import get_logger


class AppController:
    def __init__(self):
        self.logger = get_logger('AppController')
        self.file_manager = FileManager()
        self.preprocessor = ImagePreprocessor()
        self.sr_engine = SuperResolutionEngine()
        self.metrics = MetricsCalculator()
        self.saver = ResultSaver()

    def process_batch(
        self,
        settings: ProcessingSettings,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[ProcessingResult]:
        tasks = self.file_manager.build_tasks(settings)
        self.sr_engine.configure(settings.model_repo_path, settings.model_name)
        results: List[ProcessingResult] = []

        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            started = time.perf_counter()
            try:
                if progress_callback:
                    progress_callback(index, total, task.input_path.name)

                original = self.preprocessor.load(task.input_path)
                prepared = self.preprocessor.preprocess(original, convert_to_rgb=settings.convert_to_rgb)
                output = self.sr_engine.upscale(prepared, task.scale)
                self.saver.save_image(output, task.output_path)

                result = ProcessingResult(
                    input_path=str(task.input_path),
                    output_path=str(task.output_path),
                    success=True,
                    message='OK',
                    width_in=prepared.width,
                    height_in=prepared.height,
                    width_out=output.width,
                    height_out=output.height,
                    processing_time_sec=round(time.perf_counter() - started, 3),
                )

                if settings.metrics_enabled:
                    result.psnr = round(self.metrics.psnr(prepared, output), 4)
                    result.ssim = round(self.metrics.ssim(prepared, output), 4)
                    result.lpips = round(self.metrics.lpips_placeholder(prepared, output), 4)

                results.append(result)
            except Exception as exc:
                self.logger.exception('Processing error for %s', task.input_path)
                results.append(
                    ProcessingResult(
                        input_path=str(task.input_path),
                        output_path=str(task.output_path),
                        success=False,
                        message=str(exc),
                        processing_time_sec=round(time.perf_counter() - started, 3),
                    )
                )

        if settings.save_metrics and settings.output_dir is not None:
            self.saver.save_report(results, settings.output_dir / settings.report_name)
        return results
