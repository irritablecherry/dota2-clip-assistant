"""
Dota2 切片助手 - PyQt5 GUI 主界面
"""
import sys
import os
import json
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QTextEdit, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QSplitter, QFrame,
    QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from clip_detector import Dota2ClipDetector, ClipSegment
from cache_manager import VideoCacheManager


# 配置文件路径
CONFIG_FILE = Path(__file__).parent / "config.json"


def load_config() -> dict:
    """加载配置文件"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"加载配置失败：{e}")
    return {}


def save_config(config: dict):
    """保存配置文件"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存配置失败：{e}")


class AnalysisWorker(QThread):
    """视频分析工作线程 - 纯推理，不接触 UI"""
    progress = pyqtSignal(int, int)  # (current, total)
    stats_update = pyqtSignal(float, float)  # (elapsed_time, fps)
    score_update = pyqtSignal(int, int, str)  # (radiant, dire, game_time)
    log_message = pyqtSignal(str)  # 日志消息
    finished = pyqtSignal(object)  # (segments, stats)
    error = pyqtSignal(str)

    def __init__(self, detector: Dota2ClipDetector, video_path: str, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.video_path = video_path
        self.start_time = 0
        self.end_time = 0

    def run(self):
        import traceback
        self.start_time = time.time()
        try:
            total_frames = 0
            last_update_time = time.time()

            # 设置人头数识别回调
            def score_callback(radiant, dire, game_time):
                self.score_update.emit(radiant, dire, game_time)

            # 设置日志回调
            def log_callback(message):
                self.log_message.emit(message)

            self.detector.score_callback = score_callback
            self.detector.log_callback = log_callback

            def callback(current, total):
                nonlocal total_frames, last_update_time
                total_frames = total
                
                # 发送进度信号
                self.progress.emit(current, total)

                # 定期发送统计信号
                current_time = time.time()
                if current_time - last_update_time >= 0.5:
                    elapsed = current_time - self.start_time
                    fps = current / elapsed if elapsed > 0 else 0
                    self.stats_update.emit(elapsed, fps)
                    last_update_time = current_time

            # 执行分析（纯推理，不接触 UI）
            segments = self.detector.analyze_video(self.video_path, callback)

            self.end_time = time.time()
            elapsed = self.end_time - self.start_time
            fps = total_frames / elapsed if elapsed > 0 else 0

            stats = {
                'elapsed_time': elapsed,
                'total_frames': total_frames,
                'processing_fps': fps
            }

            self.finished.emit((segments, stats))
            
        except Exception as e:
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            print(f"AnalysisWorker error: {error_msg}")
            self.error.emit(error_msg)


class ExtractWorker(QThread):
    """片段提取工作线程"""
    progress = pyqtSignal(str, int, int)
    stats_update = pyqtSignal(float, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, detector: Dota2ClipDetector, video_path: str,
                 segments: list, output_dir: str, merge_output: str,
                 add_fade: bool, do_merge: bool, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.video_path = video_path
        self.segments = segments
        self.output_dir = output_dir
        self.merge_output = merge_output
        self.add_fade = add_fade
        self.do_merge = do_merge
        self.start_time = 0
        self.end_time = 0

    def run(self):
        import time
        self.start_time = time.time()
        try:
            self.total_clips = len(self.segments)
            last_update_time = time.time()
            processed_clips = 0

            def callback(clip_name, current, total):
                nonlocal last_update_time, processed_clips
                self.progress.emit(clip_name, current, total)

                if isinstance(clip_name, str) and clip_name.endswith('.mp4'):
                    if current >= total - 1:
                        processed_clips += 1
                        current_time = time.time()
                        if current_time - last_update_time >= 0.5:
                            elapsed = current_time - self.start_time
                            clips_per_sec = processed_clips / elapsed if elapsed > 0 else 0
                            self.stats_update.emit(elapsed, clips_per_sec)
                            last_update_time = current_time

            # 提取所有片段
            clip_paths = self.detector.extract_all_clips(
                self.video_path, self.segments, self.output_dir,
                callback, self.add_fade, extract_first_only=False
            )

            merged_path = ""
            if self.do_merge and clip_paths:
                # 如果只有一个片段，直接复制已提取的片段
                if len(clip_paths) == 1:
                    import shutil
                    shutil.copy2(clip_paths[0], self.merge_output)
                    merged_path = self.merge_output
                else:
                    # 多个片段才进行拼接
                    def merge_callback(clip_name, current, total):
                        self.progress.emit(clip_name, int(current), int(total))
                    self.detector.merge_clips(clip_paths, self.merge_output, callback=merge_callback)
                    merged_path = self.merge_output

            self.end_time = time.time()
            elapsed = self.end_time - self.start_time

            stats = {
                'elapsed_time': elapsed,
                'total_clips': len(clip_paths),
                'clips_per_second': len(clip_paths) / elapsed if elapsed > 0 else 0
            }

            result = (clip_paths, merged_path)
            self.finished.emit((result, stats))
            
        except Exception as e:
            self.error.emit(str(e))


class VideoInfoPanel(QGroupBox):
    """视频信息面板"""

    def __init__(self):
        super().__init__("📹 视频信息")
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.file_label = QLabel("-")
        self.duration_label = QLabel("-")
        self.resolution_label = QLabel("-")
        self.fps_label = QLabel("-")
        self.frames_label = QLabel("-")

        layout.addRow("文件:", self.file_label)
        layout.addRow("时长:", self.duration_label)
        layout.addRow("分辨率:", self.resolution_label)
        layout.addRow("帧率:", self.fps_label)
        layout.addRow("总帧数:", self.frames_label)

        self.setLayout(layout)

    def update_info(self, info: dict):
        if not info:
            return

        self.file_label.setText(Path(info.get('file', '-')).name)
        self.duration_label.setText(f"{info.get('duration', 0):.2f} 秒")
        self.resolution_label.setText(f"{info.get('width', 0)} x {info.get('height', 0)}")
        self.fps_label.setText(f"{info.get('fps', 0):.2f}")
        self.frames_label.setText(f"{info.get('total_frames', 0)}")


class StatsPanel(QGroupBox):
    """处理统计面板"""

    def __init__(self):
        super().__init__("⏱️ 处理统计")
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout()

        self.elapsed_label = QLabel("-")
        self.elapsed_label.setStyleSheet("font-weight: bold; color: #2196F3;")

        self.processing_fps_label = QLabel("-")
        self.processing_fps_label.setStyleSheet("font-weight: bold; color: #4CAF50;")

        self.score_label = QLabel("-")
        self.score_label.setStyleSheet("font-weight: bold; color: #FF5722; font-size: 14px;")

        self.game_time_label = QLabel("-")
        self.game_time_label.setStyleSheet("font-weight: bold; color: #9C27B0;")

        layout.addRow("处理用时:", self.elapsed_label)
        layout.addRow("处理帧率:", self.processing_fps_label)
        layout.addRow("当前人头比:", self.score_label)
        layout.addRow("当前游戏时间:", self.game_time_label)

        self.setLayout(layout)

    def update_stats(self, elapsed: float, total_frames: int, processing_fps: float):
        if elapsed > 0:
            if elapsed < 60:
                time_str = f"{elapsed:.2f} 秒"
            else:
                mins = int(elapsed // 60)
                secs = elapsed % 60
                time_str = f"{mins} 分 {secs:.1f} 秒"
            self.elapsed_label.setText(time_str)
            self.processing_fps_label.setText(f"{processing_fps:.2f} FPS")

    def update_score(self, radiant: int, dire: int, game_time: str):
        self.score_label.setText(f"天灾 {radiant} : {dire} 夜魇")
        self.game_time_label.setText(game_time)


class SegmentsTable(QGroupBox):
    """高光片段列表"""

    def __init__(self):
        super().__init__("🎬 高光片段")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "序号", "类型", "描述", "开始时间", "结束时间", "置信度"
        ])

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(3, 100)
        self.table.setColumnWidth(4, 100)
        self.table.setColumnWidth(5, 80)

        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        layout.addWidget(self.table)
        self.setLayout(layout)

    def update_segments(self, segments: list):
        self.table.setRowCount(0)

        for i, seg in enumerate(segments):
            row = self.table.rowCount()
            self.table.insertRow(row)

            try:
                self.table.setItem(row, 0, QTableWidgetItem(str(i + 1)))
                self.table.setItem(row, 1, QTableWidgetItem(getattr(seg, 'clip_type', 'unknown')))
                self.table.setItem(row, 2, QTableWidgetItem(getattr(seg, 'description', '无描述')))
                self.table.setItem(row, 3, QTableWidgetItem(f"{getattr(seg, 'start_time', 0):.2f}s"))
                self.table.setItem(row, 4, QTableWidgetItem(f"{getattr(seg, 'end_time', 0):.2f}s"))
                self.table.setItem(row, 5, QTableWidgetItem(f"{getattr(seg, 'confidence', 0):.2f}"))
            except Exception as e:
                print(f"[ERROR] 添加片段到表格失败：{e}")
                continue

        for row in range(self.table.rowCount()):
            try:
                conf_item = self.table.item(row, 5)
                if conf_item:
                    conf = float(conf_item.text())
                    if conf >= 0.8:
                        conf_item.setBackground(Qt.green)
                    elif conf >= 0.6:
                        conf_item.setBackground(Qt.yellow)
                    else:
                        conf_item.setBackground(Qt.gray)
            except (ValueError, AttributeError) as e:
                print(f"[ERROR] 设置置信度背景色失败：{e}")
                continue


class LogPanel(QGroupBox):
    """日志面板"""

    def __init__(self):
        super().__init__("📝 日志")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(150)

        layout.addWidget(self.log_text)
        self.setLayout(layout)

    def append_log(self, message: str):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def clear_log(self):
        self.log_text.clear()


class Dota2ClipAssistant(QMainWindow):
    """Dota2 切片助手主窗口"""

    def __init__(self):
        super().__init__()
        self.detector: Optional[Dota2ClipDetector] = None
        self.video_path: Optional[str] = None
        self.segments: list = []
        self.current_info: dict = {}
        self.cache_manager: Optional[VideoCacheManager] = None

        config = load_config()
        self.last_video_dir: str = config.get("last_video_dir", "")
        self.last_output_dir: str = config.get("last_output_dir", "")

        self.init_ui()
        self.init_detector()
        self.init_cache_manager()

    def closeEvent(self, event):
        """窗口关闭时保存配置"""
        config = {
            "last_video_dir": self.last_video_dir,
            "last_output_dir": self.last_output_dir
        }
        save_config(config)
        event.accept()

    def init_ui(self):
        """初始化 UI"""
        self.setWindowTitle("🎮 Dota2 切片助手 - 高光片段提取 v2.0")
        self.setMinimumSize(1000, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部控制区
        control_group = QGroupBox("⚙️ 控制面板")
        control_layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        self.file_path_edit = QTextEdit()
        self.file_path_edit.setPlaceholderText("请选择 Dota2 比赛视频文件...")
        self.file_path_edit.setMaximumHeight(40)
        self.file_path_edit.setReadOnly(True)

        self.select_btn = QPushButton("📁 选择视频")
        self.select_btn.clicked.connect(self.select_video)
        self.select_btn.setMaximumHeight(40)

        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.select_btn)
        control_layout.addLayout(file_layout)

        param_layout = QHBoxLayout()
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 0.99)
        self.confidence_spin.setValue(0.8)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setPrefix("置信度阈值：")

        self.detect_interval_spin = QSpinBox()
        self.detect_interval_spin.setRange(1, 60)
        self.detect_interval_spin.setValue(30)
        self.detect_interval_spin.setPrefix("检测间隔帧数：")

        self.merge_check = QCheckBox("合并重叠片段")
        self.merge_check.setChecked(True)

        param_layout.addWidget(QLabel("检测参数:"))
        param_layout.addWidget(self.confidence_spin)
        param_layout.addWidget(self.detect_interval_spin)
        param_layout.addWidget(self.merge_check)
        param_layout.addStretch()
        control_layout.addLayout(param_layout)

        effect_layout = QHBoxLayout()
        self.fade_check = QCheckBox("添加淡入淡出转场")
        self.fade_check.setChecked(True)
        self.merge_clips_check = QCheckBox("拼接所有片段为完整视频")
        self.merge_clips_check.setChecked(True)

        effect_layout.addWidget(self.fade_check)
        effect_layout.addWidget(self.merge_clips_check)
        effect_layout.addStretch()
        control_layout.addLayout(effect_layout)

        ocr_layout = QHBoxLayout()
        self.ocr_check = QCheckBox("启用人头数变化检测")
        self.ocr_check.setChecked(True)
        self.ocr_check.setToolTip("通过识别比分区域的人头数变化来捕捉团战时刻（EasyOCR）")
        ocr_layout.addWidget(self.ocr_check)
        ocr_layout.addStretch()
        control_layout.addLayout(ocr_layout)

        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("🔍 分析视频")
        self.analyze_btn.clicked.connect(self.analyze_video)
        self.analyze_btn.setEnabled(False)

        self.extract_btn = QPushButton("✂️ 提取片段")
        self.extract_btn.clicked.connect(self.extract_clips)
        self.extract_btn.setEnabled(False)

        self.clear_btn = QPushButton("🗑️ 清空")
        self.clear_btn.clicked.connect(self.clear_all)

        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addWidget(self.extract_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        control_layout.addLayout(btn_layout)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        splitter = QSplitter(Qt.Vertical)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        info_stats_layout = QHBoxLayout()
        self.info_panel = VideoInfoPanel()
        self.stats_panel = StatsPanel()
        info_stats_layout.addWidget(self.info_panel)
        info_stats_layout.addWidget(self.stats_panel)
        left_layout.addLayout(info_stats_layout)
        
        self.segments_table = SegmentsTable()
        left_layout.addWidget(self.segments_table)

        self.log_panel = LogPanel()

        splitter.addWidget(left_widget)
        splitter.addWidget(self.log_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self.statusBar().showMessage("就绪 - 请选择视频文件开始")

    def init_detector(self):
        """初始化检测器"""
        model_path = Path(__file__).parent / "model" / "best.pt"

        if not model_path.exists():
            QMessageBox.critical(self, "错误", f"未找到模型文件：{model_path}")
            sys.exit(1)

        try:
            device = 'cpu'
            try:
                import torch
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    device = 'cuda'
                    gpu_name = torch.cuda.get_device_name(0)
                    self.log_panel.append_log(f"🎮 检测到 GPU: {gpu_name}")
                else:
                    self.log_panel.append_log("ℹ️ 未检测到 GPU，使用 CPU 模式")
            except (ImportError, OSError) as e:
                self.log_panel.append_log(f"ℹ️ PyTorch 初始化提示：{str(e)[:50]}")
                self.log_panel.append_log("ℹ️ 未检测到 GPU，使用 CPU 模式")

            detect_interval = 30
            if hasattr(self, 'detect_interval_spin'):
                detect_interval = self.detect_interval_spin.value()

            self.detector = Dota2ClipDetector(
                str(model_path),
                confidence_threshold=0.5,
                use_ocr=True,
                device=device,
                detect_interval=detect_interval
            )
            self.log_panel.append_log(f"✅ 模型加载成功：{model_path.name}")
            self.log_panel.append_log(f"📊 检测间隔：每 {detect_interval} 帧检测一次人头数")
            self.log_panel.append_log("🔍 人头数检测已就绪（EasyOCR）")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败：{str(e)}")
            sys.exit(1)

    def init_cache_manager(self):
        """初始化缓存管理器"""
        try:
            self.cache_manager = VideoCacheManager()
            self.log_panel.append_log("💾 缓存管理器已就绪")
        except Exception as e:
            self.log_panel.append_log(f"⚠️ 缓存管理器初始化失败：{e}")
            self.cache_manager = None

    def select_video(self):
        """选择视频文件"""
        initial_dir = self.last_video_dir if self.last_video_dir else os.path.expanduser("~")

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", initial_dir,
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv);;所有文件 (*.*)"
        )

        if file_path:
            self.video_path = file_path
            self.file_path_edit.setText(file_path)
            self.last_video_dir = os.path.dirname(file_path)
            save_config({
                "last_video_dir": self.last_video_dir,
                "last_output_dir": self.last_output_dir
            })
            self.analyze_btn.setEnabled(True)
            self.extract_btn.setEnabled(False)
            self.statusBar().showMessage(f"已选择：{Path(file_path).name}")
            self.log_panel.append_log(f"📁 选择视频：{Path(file_path).name}")

    def analyze_video(self):
        """分析视频"""
        if not self.video_path:
            return

        # 检查缓存
        if self.cache_manager:
            cache_info = self.cache_manager.get_cache_info(self.video_path)
            if cache_info and cache_info.get('exists'):
                current_config = {
                    'confidence_threshold': self.confidence_spin.value(),
                    'use_ocr': self.ocr_check.isChecked(),
                    'detect_interval': self.detect_interval_spin.value()
                }
                
                cached_config = cache_info.get('config', {})
                config_changed = False
                changed_keys = []
                for key in ['confidence_threshold', 'use_ocr', 'detect_interval']:
                    if key in current_config and key in cached_config:
                        if current_config[key] != cached_config[key]:
                            config_changed = True
                            changed_keys.append(key)
                
                cache_time = cache_info.get('analyze_time', '未知')
                segment_count = cache_info.get('segment_count', 0)
                msg = (
                    f"发现该视频的缓存分析结果！\n\n"
                    f"分析时间：{cache_time}\n"
                    f"片段数量：{segment_count} 个\n\n"
                )
                if config_changed:
                    msg += f"⚠️ 注意：以下配置与上次不同，可能导致结果不准确：\n"
                    msg += f"   - {', '.join(changed_keys)}\n\n"
                
                msg += "您想要："
                
                dialog = QMessageBox(self)
                dialog.setIcon(QMessageBox.Information)
                dialog.setWindowTitle("发现缓存")
                dialog.setText(msg)
                
                use_cache_btn = dialog.addButton("使用缓存 (_U)", QMessageBox.YesRole)
                reanalyze_btn = dialog.addButton("重新分析 (_R)", QMessageBox.NoRole)
                cancel_btn = dialog.addButton("取消 (_C)", QMessageBox.RejectRole)
                
                dialog.setDefaultButton(use_cache_btn)
                dialog.exec_()
                
                clicked_btn = dialog.clickedButton()
                
                if clicked_btn == cancel_btn:
                    self.statusBar().showMessage("已取消分析")
                    return
                elif clicked_btn == reanalyze_btn:
                    self.cache_manager.delete_cache(self.video_path)
                    self.log_panel.append_log("🗑️ 已删除旧缓存，将重新分析")
                elif clicked_btn == use_cache_btn:
                    self.load_cached_analysis()
                    return

        self._start_analysis()

    def _start_analysis(self):
        """开始分析视频"""
        self.analyze_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("正在分析视频...")
        self.log_panel.append_log("🔍 开始分析视频...")

        self.detector.confidence_threshold = self.confidence_spin.value()
        self.detector.use_ocr = self.ocr_check.isChecked()
        self.detector.detect_interval = self.detect_interval_spin.value()

        if self.detector.use_ocr:
            self.log_panel.append_log("   📊 人头数检测：启用（EasyOCR）")
        else:
            self.log_panel.append_log("   📊 人头数检测：禁用")

        self.current_info = self.detector.get_video_info(self.video_path)
        self.current_info['file'] = self.video_path
        self.info_panel.update_info(self.current_info)

        self.worker = AnalysisWorker(self.detector, self.video_path, parent=self)
        self.worker.progress.connect(self.on_analysis_progress)
        self.worker.stats_update.connect(self.on_stats_update)
        self.worker.score_update.connect(self.on_score_update)
        self.worker.log_message.connect(self.on_analysis_log)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.worker.start()

    def load_cached_analysis(self):
        """从缓存加载分析结果"""
        if not self.cache_manager:
            self.log_panel.append_log("❌ 缓存管理器不可用")
            return

        self.log_panel.append_log("💾 正在加载缓存分析结果...")
        
        cache_data = self.cache_manager.load_cache(
            self.video_path,
            check_config_match=True,
            current_config={
                'confidence_threshold': self.confidence_spin.value(),
                'use_ocr': self.ocr_check.isChecked(),
                'detect_interval': self.detect_interval_spin.value()
            }
        )

        if cache_data is None:
            self.log_panel.append_log("❌ 缓存加载失败，将重新分析")
            self._start_analysis()
            return

        self.current_info = {
            'file': self.video_path,
            'duration': cache_data.duration,
            'width': cache_data.width,
            'height': cache_data.height,
            'fps': cache_data.fps,
            'total_frames': cache_data.total_frames
        }
        self.info_panel.update_info(self.current_info)

        self.segments = []
        for seg_data in cache_data.segments:
            self.segments.append(ClipSegment(
                start_frame=seg_data.start_frame,
                end_frame=seg_data.end_frame,
                start_time=seg_data.start_time,
                end_time=seg_data.end_time,
                clip_type=seg_data.clip_type,
                confidence=seg_data.confidence,
                description=seg_data.description
            ))

        self.segments_table.update_segments(self.segments)
        self.extract_btn.setEnabled(len(self.segments) > 0)
        self.analyze_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

        self.stats_panel.update_stats(0, cache_data.total_frames, 0)
        self.stats_panel.elapsed_label.setText("从缓存加载")

        self.statusBar().showMessage(f"已从缓存加载 - 发现 {len(self.segments)} 个高光片段")
        self.log_panel.append_log(f"✅ 已从缓存加载 {len(self.segments)} 个高光片段")
        self.log_panel.append_log(f"   📅 原始分析时间：{cache_data.analyze_time}")
        
        for seg in self.segments:
            self.log_panel.append_log(
                f"   🎬 {seg.description}: {seg.start_time:.2f}s ~ {seg.end_time:.2f}s "
                f"(置信度：{seg.confidence:.2f})"
            )

    def on_analysis_progress(self, current: int, total: int):
        """分析进度更新"""
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(f"分析中：{current}/{total} 帧 ({progress}%)")

    def on_stats_update(self, elapsed: float, processing_fps: float):
        """处理统计实时更新"""
        self.stats_panel.update_stats(elapsed, 0, processing_fps)

    def on_score_update(self, radiant: int, dire: int, game_time: str):
        """人头数识别实时更新"""
        self.stats_panel.update_score(radiant, dire, game_time)

        import time
        current_time = time.time()
        if not hasattr(self, '_last_score_log_time'):
            self._last_score_log_time = 0

        if hasattr(self, '_last_score'):
            old_r, old_d = self._last_score
            new_r, new_d = radiant, dire
            if new_r > old_r:
                self.log_panel.append_log(f"🔴 天灾人头变化：{old_r} → {new_r} (+{new_r - old_r})")
            if new_d > old_d:
                self.log_panel.append_log(f"🔵 夜魇人头变化：{old_d} → {new_d} (+{new_d - old_d})")

        if current_time - self._last_score_log_time >= 1.0:
            self.log_panel.append_log(f"📊 当前比分：天灾 {radiant} : {dire} 夜魇  时间：{game_time}")
            self._last_score_log_time = current_time

        self._last_score = (radiant, dire)

    def on_analysis_log(self, message: str):
        """分析日志消息"""
        self.log_panel.append_log(message)

    def on_analysis_finished(self, result: object):
        """分析完成"""
        try:
            segments, stats = result

            self.segments = segments
            try:
                self.segments_table.update_segments(segments)
            except Exception as e:
                print(f"[ERROR] 更新片段表格失败：{e}")

            self.progress_bar.setVisible(False)
            self.analyze_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            self.extract_btn.setEnabled(len(segments) > 0)

            try:
                self.stats_panel.update_stats(
                    stats['elapsed_time'],
                    stats['total_frames'],
                    stats['processing_fps']
                )
            except Exception as e:
                print(f"[ERROR] 更新统计信息失败：{e}")

            # 保存缓存
            if self.cache_manager:
                config = {
                    'confidence_threshold': self.detector.confidence_threshold,
                    'use_ocr': self.detector.use_ocr,
                    'detect_interval': self.detector.detect_interval
                }
                try:
                    self.cache_manager.save_cache(
                        self.video_path,
                        segments,
                        config,
                        self.current_info
                    )
                    self.log_panel.append_log("💾 已保存分析结果到缓存")
                except Exception as e:
                    print(f"[CACHE] 保存缓存失败：{e}")

            self.statusBar().showMessage(f"分析完成 - 发现 {len(segments)} 个高光片段")
            self.log_panel.append_log(f"✅ 分析完成，发现 {len(segments)} 个高光片段:")
            self.log_panel.append_log(f"   ⏱️ 处理用时：{stats['elapsed_time']:.2f} 秒 ({stats['processing_fps']:.2f} FPS)")

            for seg in segments:
                try:
                    self.log_panel.append_log(
                        f"   🎬 {seg.description}: {seg.start_time:.2f}s ~ {seg.end_time:.2f}s "
                        f"(置信度：{seg.confidence:.2f})"
                    )
                except Exception as e:
                    print(f"[ERROR] 添加日志失败：{e}")
                    continue

            if not segments:
                self.log_panel.append_log("⚠️ 未检测到任何高光片段")

        except Exception as e:
            import traceback
            print(f"[ERROR] on_analysis_finished 方法出错：{e}")
            print(f"[ERROR] {traceback.format_exc()}")
            self.statusBar().showMessage("分析完成时发生错误")
            self.log_panel.append_log(f"❌ 分析完成时发生错误：{e}")

    def on_analysis_error(self, error: str):
        """分析出错"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

        QMessageBox.critical(self, "分析失败", error)
        self.log_panel.append_log(f"❌ 分析失败：{error}")
        self.statusBar().showMessage("分析失败")

    def extract_clips(self):
        """提取片段"""
        if not self.segments:
            return

        initial_dir = self.last_output_dir if self.last_output_dir else os.path.dirname(self.video_path)

        output_dir = QFileDialog.getExistingDirectory(
            self, "选择输出目录", initial_dir
        )

        if not output_dir:
            return

        self.last_output_dir = output_dir
        save_config({
            "last_video_dir": self.last_video_dir,
            "last_output_dir": self.last_output_dir
        })

        video_name = Path(self.video_path).stem
        merge_output = os.path.join(output_dir, f"{video_name}_highlights_merged.mp4")

        self.extract_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("正在提取片段...")

        add_fade = self.fade_check.isChecked()
        do_merge = self.merge_clips_check.isChecked()

        self.log_panel.append_log("✂️ 开始提取片段...")
        if add_fade:
            self.log_panel.append_log("   🎞️ 添加淡入淡出转场效果")
        if do_merge:
            self.log_panel.append_log(f"   🔗 拼接所有片段为：{Path(merge_output).name}")

        self.worker = ExtractWorker(
            self.detector, self.video_path,
            self.segments, output_dir, merge_output,
            add_fade, do_merge, parent=self
        )
        self.worker.progress.connect(self.on_extract_progress)
        self.worker.stats_update.connect(self.on_extract_stats_update)
        self.worker.finished.connect(self.on_extract_finished)
        self.worker.error.connect(self.on_extract_error)
        self.worker.start()

    def on_extract_progress(self, clip_name: str, current: int, total: int):
        """提取进度更新"""
        if "拼接中" in clip_name:
            progress = int((current / total) * 100) if total > 0 else 0
            self.progress_bar.setValue(progress)
            self.statusBar().showMessage(f"拼接中：{current}/{total} ({progress}%)")
        else:
            progress = int(((current) / total) * 100) if total > 0 else 0
            self.progress_bar.setValue(progress)
            self.statusBar().showMessage(f"提取中：{clip_name} ({current}/{total})")

    def on_extract_stats_update(self, elapsed: float, clips_per_second: float):
        """提取统计实时更新"""
        self.stats_panel.update_stats(elapsed, 0, clips_per_second)

    def on_extract_finished(self, result: object):
        """提取完成"""
        (clip_paths, merged_path), stats = result

        self.progress_bar.setVisible(False)
        self.extract_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

        self.statusBar().showMessage(f"提取完成 - 生成 {len(clip_paths)} 个片段")
        self.log_panel.append_log(f"✅ 提取完成，生成 {len(clip_paths)} 个片段:")
        self.log_panel.append_log(f"   ⏱️ 提取用时：{stats['elapsed_time']:.2f} 秒 ({stats['clips_per_second']:.2f} 片段/秒)")
        self.log_panel.append_log(f"   📂 输出目录：{self.worker.output_dir}")

        for path in clip_paths:
            self.log_panel.append_log(f"   📁 {path}")

        if merged_path:
            self.log_panel.append_log(f"🔗 拼接视频：{merged_path}")

        output_dir = self.worker.output_dir

        QMessageBox.information(
            self, "完成",
            f"成功提取 {len(clip_paths)} 个高光片段!\n"
            f"输出目录：{output_dir}\n"
            f"{'拼接视频：' + Path(merged_path).name if merged_path else ''}"
        )

    def on_extract_error(self, error: str):
        """提取出错"""
        self.progress_bar.setVisible(False)
        self.extract_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

        QMessageBox.critical(self, "提取失败", error)
        self.log_panel.append_log(f"❌ 提取失败：{error}")
        self.statusBar().showMessage("提取失败")

    def clear_all(self):
        """清空所有"""
        self.video_path = None
        self.segments = []
        self.current_info = {}

        self.file_path_edit.clear()
        self.info_panel.update_info({})
        self.segments_table.update_segments([])
        self.stats_panel.update_score(0, 0, "0:00")
        self.log_panel.clear_log()

        self.analyze_btn.setEnabled(False)
        self.extract_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

        self.statusBar().showMessage("已清空 - 请选择新的视频文件")
        self.log_panel.append_log("🗑️ 已清空所有数据")


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = Dota2ClipAssistant()
    window.show()
    sys.exit(app.exec_())
