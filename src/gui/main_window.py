from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)
from PIL import Image

from config.settings import ProcessingSettings
from controllers.worker import ProcessingWorker
from gui.widgets.image_preview import ImagePreview
from utils.image_utils import pil_to_qpixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._last_results = []
        self.supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        self.setWindowTitle('Система улучшения изображений на базе Super-Resolution')
        self.resize(1400, 900)
        self._build_actions()
        self._build_ui()
        self._apply_styles()

    def _build_actions(self):
        self.action_open_input = QAction('Выбрать входную папку', self)
        self.action_open_input.triggered.connect(self.select_input_dir)

        self.action_open_output = QAction('Выбрать выходную папку', self)
        self.action_open_output.triggered.connect(self.select_output_dir)

        self.action_open_repo = QAction('Репозиторий модели super-resolution', self)
        self.action_open_repo.triggered.connect(self.select_repo_dir)

        self.action_run = QAction('Запустить обработку', self)
        self.action_run.triggered.connect(self.run_processing)

        self.action_exit = QAction('Выход', self)
        self.action_exit.triggered.connect(self.close)

    def _build_ui(self):
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        header = self._create_header_panel()
        settings_panel = self._create_settings_panel()
        summary_panel = self._create_summary_panel()
        workspace = self._create_workspace()

        root_layout.addWidget(header)
        root_layout.addWidget(settings_panel)
        root_layout.addWidget(summary_panel)
        root_layout.addWidget(workspace, 1)

        self.setCentralWidget(root)

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu('Файл')
        file_menu.addAction(self.action_open_input)
        file_menu.addAction(self.action_open_output)
        file_menu.addAction(self.action_open_repo)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        run_menu = menu.addMenu('Обработка')
        run_menu.addAction(self.action_run)

    def _build_toolbar(self):
        toolbar = QToolBar('Основная панель')
        toolbar.setMovable(False)
        toolbar.addAction(self.action_open_input)
        toolbar.addAction(self.action_open_output)
        toolbar.addAction(self.action_open_repo)
        toolbar.addSeparator()
        toolbar.addAction(self.action_run)
        self.addToolBar(toolbar)

    def _build_statusbar(self):
        status = QStatusBar()
        self.status_label = QLabel('Готово к работе')
        status.addWidget(self.status_label)
        self.setStatusBar(status)

    def _create_header_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName('HeaderPanel')
        layout = QVBoxLayout(frame)
        title = QLabel('Повышение качества изображений')
        title.setObjectName('TitleLabel')

        layout.addWidget(title)

        return frame

    def _create_settings_panel(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        paths_group = QGroupBox('Пути и источники данных')
        paths_form = QGridLayout(paths_group)

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.repo_edit = QLineEdit()

        self.btn_input = QPushButton('Обзор...')
        self.btn_output = QPushButton('Обзор...')
        self.btn_repo = QPushButton('Обзор...')

        self.btn_input.clicked.connect(self.select_input_dir)
        self.btn_output.clicked.connect(self.select_output_dir)
        self.btn_repo.clicked.connect(self.select_repo_dir)

        paths_form.addWidget(QLabel('Входной каталог'), 0, 0)
        paths_form.addWidget(self.input_edit, 0, 1)
        paths_form.addWidget(self.btn_input, 0, 2)
        paths_form.addWidget(QLabel('Выходной каталог'), 1, 0)
        paths_form.addWidget(self.output_edit, 1, 1)
        paths_form.addWidget(self.btn_output, 1, 2)
        paths_form.addWidget(QLabel('Репозиторий модели super-resolution'), 2, 0)
        paths_form.addWidget(self.repo_edit, 2, 1)
        paths_form.addWidget(self.btn_repo, 2, 2)

        options_group = QGroupBox('Параметры обработки')
        options_form = QFormLayout(options_group)

        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 8)
        self.scale_spin.setValue(4)
        self.metrics_check = QCheckBox('Вычислять метрики качества')
        self.metrics_check.setChecked(True)
        self.overwrite_check = QCheckBox('Перезаписывать существующие результаты')
        self.overwrite_check.setChecked(True)
        self.rgb_check = QCheckBox('Приводить изображения к RGB')
        self.rgb_check.setChecked(True)

        options_form.addRow('Масштаб увеличения', self.scale_spin)
        options_form.addRow('', self.metrics_check)
        options_form.addRow('', self.overwrite_check)

        layout.addWidget(paths_group, 2)
        layout.addWidget(options_group, 1)
        return container

    def _create_summary_panel(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.summary_total = self._create_stat_card('Всего файлов', '0')
        self.summary_success = self._create_stat_card('Успешно', '0')
        self.summary_failed = self._create_stat_card('Ошибок', '0')
        self.summary_current = self._create_stat_card('Текущий файл', '—')

        layout.addWidget(self.summary_total)
        layout.addWidget(self.summary_success)
        layout.addWidget(self.summary_failed)
        layout.addWidget(self.summary_current, 2)
        return container

    def _create_stat_card(self, title: str, value: str) -> QWidget:
        card = QFrame()
        card.setObjectName('StatCard')
        layout = QVBoxLayout(card)
        title_label = QLabel(title)
        title_label.setObjectName('StatTitle')
        value_label = QLabel(value)
        value_label.setObjectName('StatValue')
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        card.value_label = value_label
        return card

    def _create_workspace(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        run_bar = QWidget()
        run_layout = QHBoxLayout(run_bar)
        run_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_run = QPushButton('Запустить обработку')
        self.btn_run.setObjectName('RunButton')
        self.btn_run.clicked.connect(self.run_processing)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.progress, 1)

        self.results_table = QTableWidget(0, 8)
        self.results_table.setHorizontalHeaderLabels([
            'Файл', 'Статус', 'Вход', 'Выход', 'Время, сек', 'PSNR', 'SSIM', 'LPIPS'
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self.load_selected_preview)

        log_group = QGroupBox('Журнал обработки')
        log_layout = QVBoxLayout(log_group)
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        log_layout.addWidget(self.status_log)

        left_layout.addWidget(run_bar)
        left_layout.addWidget(self.results_table, 3)
        left_layout.addWidget(log_group, 2)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        preview_group = QGroupBox('Предпросмотр изображений')
        preview_layout = QVBoxLayout(preview_group)
        self.before_preview = ImagePreview('Исходное изображение')
        self.after_preview = ImagePreview('Результат обработки')
        preview_layout.addWidget(self.before_preview)
        preview_layout.addWidget(self.after_preview)

        details_group = QGroupBox('Информация о выбранном результате')
        details_layout = QFormLayout(details_group)
        self.detail_name = QLabel('—')
        self.detail_status = QLabel('—')
        self.detail_size_in = QLabel('—')
        self.detail_size_out = QLabel('—')
        self.detail_metrics = QLabel('—')
        self.detail_metrics.setWordWrap(True)
        details_layout.addRow('Файл', self.detail_name)
        details_layout.addRow('Статус', self.detail_status)
        details_layout.addRow('Размер входа', self.detail_size_in)
        details_layout.addRow('Размер выхода', self.detail_size_out)
        details_layout.addRow('Метрики', self.detail_metrics)

        right_layout.addWidget(preview_group, 3)
        right_layout.addWidget(details_group, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([900, 420])
        return splitter

    def _apply_styles(self):
        self.setStyleSheet('''
            QMainWindow { background: #f4f6f8; }
            QGroupBox {
                border: 1px solid #d8dde3;
                border-radius: 10px;
                margin-top: 10px;
                font-weight: 600;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            #HeaderPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #114b5f, stop:1 #1a759f);
                border-radius: 14px;
                padding: 12px;
            }
            #TitleLabel {
                color: white;
                font-size: 24px;
                font-weight: 700;
            }
            #SubtitleLabel {
                color: #e8f1f2;
                font-size: 13px;
            }
            #StatCard {
                background: #ffffff;
                border: 1px solid #d8dde3;
                border-radius: 12px;
                padding: 10px;
            }
            #StatTitle { color: #6b7280; font-size: 12px; }
            #StatValue { color: #111827; font-size: 20px; font-weight: 700; }
            #RunButton {
                background: #0f766e;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
                font-weight: 700;
            }
            #RunButton:hover { background: #0d9488; }
            QPushButton {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 8px 12px;
            }
            QPushButton:hover { background: #f8fafc; }
            QLineEdit, QSpinBox, QTextEdit, QTableWidget {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 6px;
            }
            QHeaderView::section {
                background: #eef2f7;
                border: none;
                border-bottom: 1px solid #d8dde3;
                padding: 8px;
                font-weight: 600;
            }
        ''')

    def select_input_dir(self):
        path = QFileDialog.getExistingDirectory(self, 'Выберите входной каталог')
        if path:
            self.input_edit.setText(path)
            self.status_label.setText('Выбран входной каталог')

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, 'Выберите выходной каталог')
        if path:
            self.output_edit.setText(path)
            self.status_label.setText('Выбран выходной каталог')

    def select_repo_dir(self):
        path = QFileDialog.getExistingDirectory(self, 'Выберите каталог репозитория Real-ESRGAN')
        if path:
            self.repo_edit.setText(path)
            self.status_label.setText('Указан путь к репозиторию модели')

    def _collect_settings(self) -> ProcessingSettings:
        return ProcessingSettings(
            input_dir=Path(self.input_edit.text()) if self.input_edit.text() else None,
            output_dir=Path(self.output_edit.text()) if self.output_edit.text() else None,
            scale=self.scale_spin.value(),
            metrics_enabled=self.metrics_check.isChecked(),
            save_metrics=True,
            overwrite=self.overwrite_check.isChecked(),
            convert_to_rgb=self.rgb_check.isChecked(),
            model_repo_path=Path(self.repo_edit.text()) if self.repo_edit.text() else None,
        )

    def run_processing(self):
        try:
            settings = self._collect_settings()

            # 1. Проверка: выбраны входной и выходной каталоги
            if settings.input_dir is None or settings.output_dir is None:
                QMessageBox.warning(
                    self,
                    'Каталоги не выбраны',
                    'Необходимо выбрать входной и выходной каталоги перед запуском обработки.'
                )
                return

            input_dir = settings.input_dir
            if not input_dir.exists() or not input_dir.is_dir():
                QMessageBox.warning(
                    self,
                    'Некорректный каталог',
                    'Указанный входной каталог не существует или недоступен.'
                )
                return

            # 2. Собираем файлы из входного каталога
            all_files = list(input_dir.iterdir())
            if not all_files:
                QMessageBox.warning(
                    self,
                    'Пустой каталог',
                    'Выбранный входной каталог не содержит файлов для обработки.'
                )
                return

            image_files = []
            unsupported_files = []

            for f in all_files:
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in self.supported_ext:
                    image_files.append(f)
                else:
                    unsupported_files.append(f)

            # 3. Нет ни одного подходящего изображения
            if not image_files:
                QMessageBox.warning(
                    self,
                    'Нет подходящих файлов',
                    'В выбранном каталоге отсутствуют изображения поддерживаемых форматов: '
                    f"{', '.join(sorted(self.supported_ext))}."
                )
                return

            # 4. Есть неподдерживаемые файлы — предупредить, но продолжить
            if unsupported_files:
                QMessageBox.information(
                    self,
                    'Неподдерживаемые файлы',
                    'В выбранном каталоге обнаружены файлы неподдерживаемых форматов.\n'
                    'Они будут пропущены при обработке.'
                )

            # 5. Запуск обработки (как и раньше)
            self._set_controls_enabled(False)
            self.status_log.clear()
            self.results_table.setRowCount(0)
            self.before_preview.set_preview(None)
            self.after_preview.set_preview(None)
            self._update_details(None)
            self._update_summary(total=0, success=0, failed=0, current='—')
            self.progress.setValue(0)
            self.status_label.setText('Запущена пакетная обработка')

            # ВАЖНО: передаём settings как раньше — список файлов, скорее всего,
            # формируется внутри ProcessingWorker на основе settings.input_dir.
            # Если там нужна фильтрация, можно будет использовать supported_ext и там тоже.
            self.worker = ProcessingWorker(settings)
            self.worker.progress_changed.connect(self.on_progress)
            self.worker.finished_success.connect(self.on_finished)
            self.worker.failed.connect(self.on_failed)
            self.worker.start()
        except Exception as exc:
            self._set_controls_enabled(True)
            QMessageBox.critical(self, 'Ошибка', str(exc))

    def on_progress(self, current: int, total: int, filename: str):
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(current)
        self.status_log.append(f'Обработка: {filename} ({current}/{total})')
        self._update_summary(total=total, success=self._count_success(), failed=self._count_failed(), current=filename)
        self.status_label.setText(f'Обрабатывается файл: {filename}')

    def on_finished(self, results):
        self._last_results = results
        self._fill_results(results)
        self._set_controls_enabled(True)
        success = sum(1 for r in results if r.success)
        failed = len(results) - success
        self._update_summary(total=len(results), success=success, failed=failed, current='Завершено')
        self.status_label.setText('Пакетная обработка завершена')
        QMessageBox.information(self, 'Готово', 'Пакетная обработка завершена')

    def on_failed(self, message: str):
        self._set_controls_enabled(True)
        self.status_label.setText('Ошибка обработки')
        QMessageBox.critical(self, 'Ошибка', message)

    def _fill_results(self, results):
        self.results_table.setRowCount(0)
        for row_idx, item in enumerate(results):
            self.results_table.insertRow(row_idx)
            file_item = QTableWidgetItem(Path(item.input_path).name)
            file_item.setData(Qt.ItemDataRole.UserRole, item)
            self.results_table.setItem(row_idx, 0, file_item)
            self.results_table.setItem(row_idx, 1, QTableWidgetItem('OK' if item.success else item.message))
            self.results_table.setItem(row_idx, 2, QTableWidgetItem(f'{item.width_in}x{item.height_in}' if item.width_in else '-'))
            self.results_table.setItem(row_idx, 3, QTableWidgetItem(f'{item.width_out}x{item.height_out}' if item.width_out else '-'))
            self.results_table.setItem(row_idx, 4, QTableWidgetItem(str(item.processing_time_sec) if item.processing_time_sec is not None else '-'))
            self.results_table.setItem(row_idx, 5, QTableWidgetItem(str(item.psnr) if item.psnr is not None else '-'))
            self.results_table.setItem(row_idx, 6, QTableWidgetItem(str(item.ssim) if item.ssim is not None else '-'))
            self.results_table.setItem(row_idx, 7, QTableWidgetItem(str(item.lpips) if item.lpips is not None else '-'))

        if results:
            self.results_table.selectRow(0)

    def load_selected_preview(self):
        row = self.results_table.currentRow()
        if row < 0:
            return
        item = self.results_table.item(row, 0)
        if item is None:
            return
        result = item.data(Qt.ItemDataRole.UserRole)
        try:
            self.before_preview.set_preview(pil_to_qpixmap(Image.open(result.input_path)))
            if result.success and Path(result.output_path).exists():
                self.after_preview.set_preview(pil_to_qpixmap(Image.open(result.output_path)))
            else:
                self.after_preview.set_preview(None)
            self._update_details(result)
        except Exception as exc:
            self.status_log.append(f'Ошибка предпросмотра: {exc}')

    def _update_details(self, result):
        if result is None:
            self.detail_name.setText('—')
            self.detail_status.setText('—')
            self.detail_size_in.setText('—')
            self.detail_size_out.setText('—')
            self.detail_metrics.setText('—')
            return

        self.detail_name.setText(Path(result.input_path).name)
        self.detail_status.setText('Успешно' if result.success else result.message)
        self.detail_size_in.setText(f'{result.width_in}x{result.height_in}' if result.width_in else '—')
        self.detail_size_out.setText(f'{result.width_out}x{result.height_out}' if result.width_out else '—')
        metrics_text = f'PSNR: {result.psnr}; SSIM: {result.ssim}; LPIPS: {result.lpips}'
        self.detail_metrics.setText(metrics_text)

    def _update_summary(self, total: int, success: int, failed: int, current: str):
        self.summary_total.value_label.setText(str(total))
        self.summary_success.value_label.setText(str(success))
        self.summary_failed.value_label.setText(str(failed))
        self.summary_current.value_label.setText(current)

    def _count_success(self) -> int:
        return sum(1 for r in self._last_results if r.success)

    def _count_failed(self) -> int:
        return sum(1 for r in self._last_results if not r.success)

    def _set_controls_enabled(self, enabled: bool):
        for widget in [
            self.input_edit,
            self.output_edit,
            self.repo_edit,
            self.scale_spin,
            self.metrics_check,
            self.overwrite_check,
            self.rgb_check,
            self.btn_input,
            self.btn_output,
            self.btn_repo,
            self.btn_run,
        ]:
            widget.setEnabled(enabled)
        self.action_run.setEnabled(enabled)


def run_standalone():
    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
