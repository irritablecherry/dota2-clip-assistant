"""
Dota2 视频高光片段检测核心模块
使用 YOLOv8 模型识别关键事件并提取高光片段

v3.1 - 添加 FFmpeg 视频处理支持
"""
import os
import cv2
import time
import numpy as np
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ultralytics import YOLO
import av

from score_ocr import ScoreOCRDetector, KillHighlightDetector


def check_ffmpeg_installed() -> bool:
    """检查 FFmpeg 是否已安装"""
    try:
        # 尝试执行 ffmpeg -version 命令
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def get_ffmpeg_version() -> str:
    """获取 FFmpeg 版本信息"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        if result.returncode == 0:
            # 返回第一行（通常是版本信息）
            return result.stdout.split('\n')[0].strip()
    except Exception:
        pass
    return ""


@dataclass
class ClipSegment:
    """视频片段段"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    clip_type: str
    description: str
    parent_event: str = ""  # 所属的原始事件类型


@dataclass
class CutPoint:
    """切断点 - 检测到 replay/paused 的位置"""
    frame: int
    cut_type: str  # 'replay' or 'paused'
    confidence: float


@dataclass
class InvalidRange:
    """无效时间区间"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    range_type: str  # 'replay', 'paused', 'no_score'
    confidence: float


class Dota2ClipDetector:
    """Dota2 高光片段检测器"""

    # 定义需要识别的目标类别
    TARGET_CLASSES = {
        'score_area': '比分区域',
        'fog': '开雾',
        'replay': '回放',
        'paused': '暂停',
        'victory': '胜利',
        'logo': '结束 Logo'
    }

    # 高光片段配置
    HIGHLIGHT_CONFIG = {
        'fog': {
            'pre_seconds': 5,
            'post_seconds': 10,
            'priority': 1,
            'description': '开雾时刻'
        },
        'replay': {
            'pre_seconds': 0,
            'post_seconds': 0,
            'priority': 5,
            'description': '比赛回放（自动跳过）'
        },
        'paused': {
            'pre_seconds': 0,
            'post_seconds': 0,
            'priority': 5,
            'description': '比赛暂停（自动跳过）'
        },
        'victory': {
            'pre_seconds': 10,
            'post_seconds': 30,
            'priority': 0,
            'description': '胜利时刻'
        },
        'logo': {
            'pre_seconds': 0,
            'post_seconds': 5,
            'priority': 4,
            'description': '结束画面'
        },
        'score_area': {
            'pre_seconds': 2,
            'post_seconds': 5,
            'priority': 3,
            'description': '比分区域'
        },
        'kill': {
            'pre_seconds': 20,
            'post_seconds': 10,
            'priority': 2,
            'description': '人头变化'
        }
    }

    # 无效区间配置
    INVALID_RANGE_CONFIG = {
        'replay': {
            'expand_pre_seconds': 1.0,    # 向前扩展 1 秒
            'expand_post_seconds': 1.0,   # 向后扩展 1 秒
        },
        'paused': {
            'expand_pre_seconds': 0.5,    # 向前扩展 0.5 秒
            'expand_post_seconds': 0.5,   # 向后扩展 0.5 秒
        },
        'no_score': {
            'expand_pre_seconds': 0.0,
            'expand_post_seconds': 0.0,
        }
    }

    MIN_SEGMENT_DURATION = 3.0  # 最小片段时长（秒）

    def __init__(self, model_path: str, confidence_threshold: float = 0.8,
                 use_ocr: bool = True, device: str = 'auto',
                 detect_interval: int = 30):
        """
        初始化检测器

        Args:
            model_path: YOLOv8 模型路径
            confidence_threshold: 置信度阈值
            use_ocr: 是否启用 OCR 人头数检测
            device: 推理设备 ('cuda', 'cpu', 'auto')
            detect_interval: OCR 检测间隔帧数（默认 30 帧）
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_ocr = use_ocr
        self.device = device
        self.detect_interval = detect_interval  # 检测间隔帧数
        self.model = YOLO(model_path)

        # 人头数识别回调
        self.score_callback = None
        
        # 日志回调
        self.log_callback = None

        # 自动选择设备
        if self.device == 'auto':
            try:
                import torch
                # 检查是否有可用的 GPU 且安装了 GPU 版 PyTorch
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    self.device = 'cuda'
                else:
                    self.device = 'cpu'
            except ImportError:
                self.device = 'cpu'

        # 将模型移动到指定设备
        if self.device == 'cuda':
            try:
                self.model.to('cuda')
                # print(f"✅ YOLO 模型已加载到 GPU")
            except Exception as e:
                # print(f"⚠️ GPU 加载失败：{e}，使用 CPU 模式")
                self.device = 'cpu'

        if self.device == 'cpu':
            pass  # print(f"ℹ️ YOLO 模型使用 CPU 模式")

        self.fps = 30

        # 初始化 OCR 检测器（使用 EasyOCR）
        if self.use_ocr:
            self.ocr_detector = ScoreOCRDetector(use_ocr=True)
        else:
            self.ocr_detector = None

    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """检测单帧中的目标"""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []

        if results[0].boxes is not None:
            boxes = results[0].boxes
            # 批量转换 bbox 数据，减少 GPU 到 CPU 的传输次数
            all_boxes = boxes.xyxy.cpu().numpy()
            all_cls = boxes.cls.cpu().numpy()
            all_conf = boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                cls_id = int(all_cls[i])
                conf = float(all_conf[i])
                bbox = all_boxes[i]
                class_name = self.model.names[cls_id]

                detections.append({
                    'class': class_name,
                    'class_id': cls_id,
                    'confidence': conf,
                    'bbox': bbox,
                    'description': self.TARGET_CLASSES.get(class_name, class_name)
                })

        return detections

    def analyze_video(self, video_path: str, callback=None) -> List[ClipSegment]:
        """分析整个视频，检测高光片段"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_detections: Dict[int, List[Dict]] = {}
        cut_points: List[CutPoint] = []  # 存储切断点
        kill_events: List[dict] = []  # 存储人头变化事件
        all_events: List[dict] = []  # 所有事件 人头变化和开雾
        victory_events: List[dict] = []  # 存储胜利画面事件
        no_score_frames: List[int] = []  # 未识别到比分区域的帧

        # 初始化 OCR 检测器（如果需要）
        if self.use_ocr and self.ocr_detector:
            self.ocr_detector.reset()
            # 注意：不再使用固定 ROI，改为每帧使用 YOLO 检测比分区域

        frame_idx = 0
        last_callback_score = None  # 记录上次回调的人头数，避免频繁回调
        callback_interval = 30  # 至少间隔 30 帧（1 秒@30fps）才回调一次
        fog_status = None
        fog_start_frame = None
        fog_config = self.HIGHLIGHT_CONFIG['fog']
        kill_config = self.HIGHLIGHT_CONFIG['kill']
        
        # replay/paused 状态跟踪
        replay_status = None  # None, 'START'
        replay_start_frame = None
        replay_end_buffer_frame = None  # replay 结束后的缓冲帧
        paused_status = None  # None, 'START'
        paused_start_frame = None
        paused_end_buffer_frame = None  # paused 结束后的缓冲帧

        # 比分区域检测配置
        check_score_area = self.use_ocr and self.ocr_detector
        no_score_threshold = int(2.0 * self.fps)  # 连续 2 秒无比分才视为无效
        consecutive_no_score_count = 0

        # 初始化上一次检测结果（用于间隔检测时的复用）
        last_detections = []

        # 计算最后 5 秒的起始帧（用于更密集的检测）
        # 只有当检测间隔 > 5 帧时才需要特殊处理，否则使用原始间隔即可
        last_5_seconds_start = None
        if self.detect_interval > 5:
            last_5_seconds_start = max(0, total_frames - int(5.0 * self.fps))

        start_time = time.time()
        if last_5_seconds_start is not None:
            print(f"[DEBUG] 开始分析视频，总帧数：{total_frames}, FPS: {self.fps}, 检测间隔：{self.detect_interval}帧")
            print(f"[DEBUG] 最后 5 秒起始帧：{last_5_seconds_start} ({last_5_seconds_start/self.fps:.2f}s)，该区域使用每 5 帧检测 1 次")
        else:
            print(f"[DEBUG] 开始分析视频，总帧数：{total_frames}, FPS: {self.fps}, 检测间隔：{self.detect_interval}帧")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 间隔检测
            # 如果检测间隔 > 5 帧，视频最后 5 秒使用每 5 帧检测 1 次，避免漏检决胜时刻
            if last_5_seconds_start is not None and frame_idx >= last_5_seconds_start:
                is_detect_frame = (frame_idx % 5 == 0)
            else:
                is_detect_frame = (frame_idx % self.detect_interval == 0)
            
            if is_detect_frame:
                # 执行 YOLO 检测
                detections = self.detect_frame(frame)
                last_detections = detections
            else:
                # 使用上一次的检测结果
                detections = last_detections

            # 每帧只保留每个类别置信度最高的检测结果（避免同一类别重复检测）
            if detections:
                filtered_dets = {}
                for det in detections:
                    cls = det['class']
                    if cls not in filtered_dets or det['confidence'] > filtered_dets[cls]['confidence']:
                        filtered_dets[cls] = det
                detections = list(filtered_dets.values())

            if detections:
                frame_detections[frame_idx] = detections

                # 检测 replay/paused 作为切断点，并跟踪状态（只在检测帧执行）
                if is_detect_frame:
                    has_replay = False
                    has_paused = False

                    for det in detections:
                        if det['class'] == 'replay':
                            has_replay = True
                            if replay_status is None:
                                replay_status = 'START'
                                # 动态缓冲：向前缓冲 = 检测间隔对应的时间（可能漏掉的最大时间窗口）
                                buffer_seconds = self.detect_interval / self.fps
                                replay_start_frame = frame_idx - int(buffer_seconds * self.fps)
                                replay_start_frame = max(0, replay_start_frame)  # 不能小于 0
                                log_msg = f"Replay 开始 帧 {frame_idx} ({frame_idx/self.fps:.2f}s) 置信度：{det['confidence']:.2f}, 向前缓冲 {buffer_seconds:.1f}s 至帧 {replay_start_frame}"
                                print(f"[DEBUG] {log_msg}")
                                if self.log_callback:
                                    self.log_callback(f"🎬 {log_msg}")

                        if det['class'] == 'paused':
                            has_paused = True
                            if paused_status is None:
                                paused_status = 'START'
                                # 动态缓冲：向前缓冲 = 检测间隔对应的时间（可能漏掉的最大时间窗口）
                                buffer_seconds = self.detect_interval / self.fps
                                paused_start_frame = frame_idx - int(buffer_seconds * self.fps)
                                paused_start_frame = max(0, paused_start_frame)  # 不能小于 0
                                log_msg = f"Paused 开始 帧 {frame_idx} ({frame_idx/self.fps:.2f}s) 置信度：{det['confidence']:.2f}, 向前缓冲 {buffer_seconds:.1f}s 至帧 {paused_start_frame}"
                                print(f"[DEBUG] {log_msg}")
                                if self.log_callback:
                                    self.log_callback(f"⏸️ {log_msg}")

                    # 检测 replay 结束（只在检测帧执行）
                    if not has_replay and replay_status == 'START':
                        replay_status = None
                        # 动态缓冲：向后缓冲 = 检测间隔对应的时间（可能漏掉的最大时间窗口）
                        buffer_seconds = self.detect_interval / self.fps
                        replay_end_buffer_frame = frame_idx + int(buffer_seconds * self.fps)
                        duration = (frame_idx - replay_start_frame) / self.fps
                        log_msg = f"Replay 结束 帧 {frame_idx} ({frame_idx/self.fps:.2f}s) 持续：{duration:.2f}s, 向后缓冲 {buffer_seconds:.1f}s 至帧 {replay_end_buffer_frame}"
                        print(f"[DEBUG] {log_msg}")
                        if self.log_callback:
                            self.log_callback(f"🎬 {log_msg}")

                    # 检测 paused 结束（只在检测帧执行）
                    if not has_paused and paused_status == 'START':
                        paused_status = None
                        # 动态缓冲：向后缓冲 = 检测间隔对应的时间（可能漏掉的最大时间窗口）
                        buffer_seconds = self.detect_interval / self.fps
                        paused_end_buffer_frame = frame_idx + int(buffer_seconds * self.fps)
                        duration = (frame_idx - paused_start_frame) / self.fps
                        log_msg = f"Paused 结束 帧 {frame_idx} ({frame_idx/self.fps:.2f}s) 持续：{duration:.2f}s, 向后缓冲 {buffer_seconds:.1f}s 至帧 {paused_end_buffer_frame}"
                        print(f"[DEBUG] {log_msg}")
                        if self.log_callback:
                            self.log_callback(f"⏸️ {log_msg}")

                    # 如果是 replay 或 paused 状态，添加到切断点
                    if has_replay:
                        cut_points.append(CutPoint(
                            frame=frame_idx,
                            cut_type='replay',
                            confidence=detections[0]['confidence']
                        ))
                    if has_paused:
                        cut_points.append(CutPoint(
                            frame=frame_idx,
                            cut_type='paused',
                            confidence=detections[0]['confidence']
                        ))

                    # 检测 victory 事件
                    for det in detections:
                        if det['class'] == 'victory':
                            victory_events.append({
                                'frame': frame_idx,
                                'time': frame_idx / self.fps,
                                'confidence': det['confidence']
                            })
                            # 同时将 victory 事件添加到 all_events，用于生成片段
                            victory_config = self.HIGHLIGHT_CONFIG['victory']
                            seg_start_frame = int(frame_idx - victory_config['pre_seconds'] * self.fps)
                            seg_end_frame = int(frame_idx + victory_config['post_seconds'] * self.fps)
                            all_events.append({
                                'type': 'victory',
                                'frame': frame_idx,
                                'start_frame': frame_idx,
                                'end_frame': frame_idx,
                                'start_time': frame_idx / self.fps,
                                'end_time': frame_idx / self.fps,
                                'seg_start_frame': seg_start_frame if seg_start_frame >= 0 else 0,
                                'seg_end_frame': seg_end_frame if seg_end_frame < total_frames else total_frames,
                            })
                            print(f"[DEBUG] 检测到胜利画面：帧 {frame_idx}, 置信度：{det['confidence']}")
            
            # 如果是 replay 或 paused 状态，跳过后续人头数检测逻辑
            # 检查是否在缓冲期内
            in_replay_buffer = (replay_end_buffer_frame is not None and frame_idx < replay_end_buffer_frame)
            in_paused_buffer = (paused_end_buffer_frame is not None and frame_idx < paused_end_buffer_frame)
            
            is_replay_or_paused = (replay_status == 'START' or paused_status == 'START' or 
                                   in_replay_buffer or in_paused_buffer)
            
            # 缓冲期结束检查
            if in_replay_buffer and frame_idx >= replay_end_buffer_frame:
                replay_end_buffer_frame = None
                log_msg = f"Replay 缓冲结束 帧 {frame_idx}"
                print(f"[DEBUG] {log_msg}")
                if self.log_callback:
                    self.log_callback(f"🎬 {log_msg}")
            if in_paused_buffer and frame_idx >= paused_end_buffer_frame:
                paused_end_buffer_frame = None
                log_msg = f"Paused 缓冲结束 帧 {frame_idx}"
                print(f"[DEBUG] {log_msg}")
                if self.log_callback:
                    self.log_callback(f"⏸️ {log_msg}")
            
            # 检测比分区域是否存在
            has_score_area = any(det['class'] == 'score_area' for det in detections) if detections else False

            # 以下情况视为无效帧：
            # 1. YOLO 没有检测到任何目标（detections 为空）→ 可能是黑屏/过渡画面
            # 2. 有检测结果但没有比分区域 → 无比分画面
            # 3. 处于 replay 或 paused 状态
            if is_replay_or_paused:
                # replay/paused 状态视为无效帧
                no_score_frames.append(frame_idx)
            elif not has_score_area:
                consecutive_no_score_count += 1
                no_score_frames.append(frame_idx)
            else:
                consecutive_no_score_count = 0

            # OCR 人头数检测（只在检测帧执行，使用 YOLO 检测比分区域）
            # replay/paused 状态下跳过人头数检测
            if self.use_ocr and self.ocr_detector and is_detect_frame and not is_replay_or_paused:
                # 使用 YOLO 检测比分区域
                score_area = None
                for det in detections:
                    if det['class'] == 'score_area':
                        bbox = det['bbox']
                        x1, y1, x2, y2 = map(int, bbox)
                        score_area = frame[y1:y2, x1:x2]
                        break

                if score_area is not None:

                    # 处理开雾事件
                    has_fog = False
                    for det in detections:
                        if det['class'] == 'fog':
                            has_fog = True
                            if fog_status is None:
                                fog_status = 'START'
                                fog_start_frame = frame_idx
                                print(f"[DEBUG] 开雾事件开始 置信度：{det['confidence']}")
                    # 如果监测结果中没有fog
                    if not has_fog and fog_status == 'START':
                        fog_status = None
                        seg_start_frame = int(fog_start_frame - fog_config['pre_seconds'] * self.fps)
                        seg_end_frame = int(frame_idx + fog_config['post_seconds'] * self.fps)
                        print(f"[DEBUG] 开雾事件结束")
                        all_events.append({
                            'type':'fog',
                            'frame': frame_idx,
                            'start_frame': fog_start_frame,
                            'end_frame': frame_idx,
                            'start_time': fog_start_frame / self.fps,
                            'end_time': frame_idx / self.fps,
                            'seg_start_frame': seg_start_frame if seg_start_frame >= 0 else 0,
                            'seg_end_frame': seg_end_frame if seg_end_frame < total_frames else total_frames,
                        })
                    # 使用 YOLO 检测到的比分区域进行 OCR 识别
                    score_info = self.ocr_detector.analyze_frame_with_score_area(
                        score_area, frame_idx, self.fps
                    )
                    if score_info:
                        # 获取最新的人头变化事件（检查当前帧是否有新事件）
                        events = self.ocr_detector.kill_events
                        if events:
                            # 检查最新的事件是否是当前帧产生的
                            latest_event = events[-1]
                            if latest_event.frame == frame_idx:
                                kill_events.append({
                                    'frame': frame_idx,
                                    'time': frame_idx / self.fps,
                                    'team': latest_event.team,
                                    'old_score': latest_event.old_score,
                                    'new_score': latest_event.new_score,
                                    'confidence': latest_event.confidence
                                })
                                seg_start_frame = int(frame_idx - kill_config['pre_seconds'] * self.fps)
                                seg_end_frame = int(frame_idx + kill_config['post_seconds'] * self.fps)
                                all_events.append({
                                    'type': 'kill',
                                    'frame': frame_idx,
                                    'start_frame': frame_idx,
                                    'end_frame': frame_idx,
                                    'start_time': frame_idx / self.fps,
                                    'end_time': frame_idx / self.fps,
                                    'seg_start_frame': seg_start_frame if seg_start_frame >= 0 else 0,
                                    'seg_end_frame': seg_end_frame if seg_end_frame < total_frames else total_frames,
                                })
                                print(
                                    f"[DEBUG] 人头变化：{latest_event.team} {latest_event.old_score}→{latest_event.new_score}")
                                print(
                                    f"[DEBUG] 人头变化：{score_info}")

                        # 实时输出人头数（当人头数变化时，且距离上次回调至少 30 帧）
                        current_score = (score_info.radiant_kills, score_info.dire_kills)
                        if self.score_callback and (
                                current_score != last_callback_score or
                                (last_callback_score is None) or
                                frame_idx % callback_interval == 0
                        ):
                            # 传递当前人头数和游戏时间
                            self.score_callback(
                                score_info.radiant_kills,
                                score_info.dire_kills,
                                score_info.time_str
                            )
                            last_callback_score = current_score

            if callback:
                callback(frame_idx, total_frames)

            frame_idx += 1
            
            # 每 100 帧输出一次进度
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                fps_real = frame_idx / elapsed if elapsed > 0 else 0
                print(f"[DEBUG] 分析进度：{frame_idx}/{total_frames} ({elapsed:.1f}s, {fps_real:.2f} FPS)")

        cap.release()

        # 调试日志：输出检测结果
        print(f"[DEBUG] 检测完成：{len(frame_detections)} 帧有检测结果")
        print(f"[DEBUG] 总帧数：{total_frames}, 无比分帧数：{len(no_score_frames)}")
        
        fog_count = len([event for event in all_events if event.get('type') == 'fog'])
        print(f"[DEBUG] 开雾破雾事件：{fog_count} 个")
        print(f"[DEBUG] 人头变化事件：{len(kill_events)} 个")
        print(f"[DEBUG] 胜利画面事件：{len(victory_events)} 个")
        print(f"[DEBUG] 切断点（replay/paused）: {len(cut_points)} 个")

        # 从切断点生成无效区间
        invalid_ranges = self._create_invalid_ranges(cut_points, total_frames)
        
        # 从无比分帧生成无效区间
        if no_score_frames:
            no_score_ranges = self._create_no_score_ranges(no_score_frames, total_frames)
            invalid_ranges.extend(no_score_ranges)
        
        # 按起始帧排序所有无效区间
        invalid_ranges = sorted(invalid_ranges, key=lambda r: r.start_frame)
        
        print(f"[DEBUG] 无效区间：{len(invalid_ranges)} 个")
        for ir in invalid_ranges:
            print(f"   - {ir.range_type}: {ir.start_time:.2f}s ~ {ir.end_time:.2f}s")

        # 生成高光片段并裁剪无效区间
        segments = self._generate_segments_with_crop(all_events, invalid_ranges, total_frames, victory_events)

        print(f"[DEBUG] 最终片段：{len(segments)} 个")
        for seg in segments:
            duration = seg.end_time - seg.start_time
            print(f"   - {seg.description}: {seg.start_time:.2f}s ~ {seg.end_time:.2f}s (时长：{duration:.1f}s)")

        return segments

    def _create_invalid_ranges(self, cut_points: List[CutPoint], total_frames: int) -> List[InvalidRange]:
        """从切断点创建无效区间"""
        if not cut_points:
            return []
        
        # 按帧索引排序
        sorted_points = sorted(cut_points, key=lambda p: p.frame)
        
        # 将连续的相同类型的切断点合并为区间
        ranges = []
        current_range = None
        
        for point in sorted_points:
            if current_range is None:
                current_range = {
                    'type': point.cut_type,
                    'start_frame': point.frame,
                    'end_frame': point.frame,
                    'confidence': point.confidence
                }
            elif point.cut_type == current_range['type'] and point.frame <= current_range['end_frame'] + 5:
                # 同类型且连续（允许 5 帧容差），扩展区间
                current_range['end_frame'] = point.frame
                current_range['confidence'] = max(current_range['confidence'], point.confidence)
            else:
                # 保存当前区间，开始新的区间
                ranges.append(current_range)
                current_range = {
                    'type': point.cut_type,
                    'start_frame': point.frame,
                    'end_frame': point.frame,
                    'confidence': point.confidence
                }
        
        if current_range:
            ranges.append(current_range)
        
        # 转换为 InvalidRange 对象，并扩展区间
        invalid_ranges = []
        for r in ranges:
            config = self.INVALID_RANGE_CONFIG.get(r['type'], {'expand_pre_seconds': 0.0, 'expand_post_seconds': 0.0})
            
            # 计算扩展的帧数
            expand_pre_frames = int(config['expand_pre_seconds'] * self.fps)
            expand_post_frames = int(config['expand_post_seconds'] * self.fps)
            
            # 扩展区间
            start_frame = max(0, r['start_frame'] - expand_pre_frames)
            end_frame = min(total_frames - 1, r['end_frame'] + expand_post_frames)
            
            invalid_ranges.append(InvalidRange(
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / self.fps,
                end_time=end_frame / self.fps,
                range_type=r['type'],
                confidence=r['confidence']
            ))
        
        return invalid_ranges

    def _create_no_score_ranges(self, no_score_frames: List[int], total_frames: int) -> List[InvalidRange]:
        """
        从无比分帧创建无效区间
        
        逻辑：从无比分画面出现 → 有比分画面重新出现的整个区间都设为无效帧
        前后按照 detect_interval 的频率进行缓冲
        """
        print(f"[DEBUG] _create_no_score_ranges 输入：{len(no_score_frames)} 个无比分帧")

        if not no_score_frames:
            return []

        # 计算缓冲帧数（基于检测间隔）
        buffer_frames = self.detect_interval

        # 将连续的无比分帧合并为区间（允许最多 1 秒的间断）
        gap_threshold = int(1.0 * self.fps)  # 1 秒的帧数

        raw_ranges = []
        current_start = no_score_frames[0]
        current_end = no_score_frames[0]

        for frame in no_score_frames[1:]:
            if frame - current_end <= gap_threshold:
                # 连续或间断很小，扩展当前区间
                current_end = frame
            else:
                # 间断太大，保存当前区间并开始新的区间
                if current_end - current_start >= self.fps:  # 至少 1 秒才视为无效区间
                    raw_ranges.append({
                        'start_frame': current_start,
                        'end_frame': current_end,
                        'type': 'no_score'
                    })
                current_start = frame
                current_end = frame

        # 保存最后一个区间
        if current_end - current_start >= self.fps:  # 至少 1 秒
            raw_ranges.append({
                'start_frame': current_start,
                'end_frame': current_end,
                'type': 'no_score'
            })

        print(f"[DEBUG] 合并后的原始无比分区间：{len(raw_ranges)} 个")
        for r in raw_ranges:
            print(f"   - 原始：{r['start_frame']/self.fps:.2f}s ~ {r['end_frame']/self.fps:.2f}s ({r['end_frame']-r['start_frame']+1}帧)")

        # 扩展每个区间：向前和向后各扩展 buffer_frames 帧
        # 这代表从无比分出现到比分重新出现的完整过渡区间
        invalid_ranges = []
        for r in raw_ranges:
            # 向前扩展 buffer_frames 帧（无比分出现前的缓冲）
            start_frame = max(0, r['start_frame'] - buffer_frames)
            # 向后扩展 buffer_frames 帧（比分重新出现后的缓冲）
            end_frame = min(total_frames - 1, r['end_frame'] + buffer_frames)

            invalid_ranges.append(InvalidRange(
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / self.fps,
                end_time=end_frame / self.fps,
                range_type=r['type'],
                confidence=1.0  # 无比分区域的置信度设为 1.0
            ))

        print(f"[DEBUG] 扩展后的无比分无效区间（前后缓冲{buffer_frames}帧）: {len(invalid_ranges)} 个")
        for ir in invalid_ranges:
            print(f"   - no_score: {ir.start_time:.2f}s ~ {ir.end_time:.2f}s ({ir.end_frame-ir.start_frame+1}帧)")

        return invalid_ranges

    def _generate_segments_with_crop(
        self,
        events: List[dict],
        invalid_ranges: List[InvalidRange],
        total_frames: int,
        victory_events: List[dict] = None
    ) -> List[ClipSegment]:
        """生成原始片段并裁剪无效区间"""
        
        if victory_events is None:
            victory_events = []

        # 1. 生成原始片段（前后扩展）
        raw_segments = []
        if len(events) > 0:
            # 按起始帧排序事件
            sorted_events = sorted(events, key=lambda e: e['seg_start_frame'])
            seg_start_frame = None

            for i, current_event in enumerate(sorted_events):
                if seg_start_frame is None:
                    seg_start_frame = current_event['seg_start_frame']

                # 如果是最后一个事件，直接结束当前片段
                if i == len(sorted_events) - 1:
                    raw_segments.append({
                        'type': current_event["type"],
                        'start_frame': seg_start_frame,
                        'end_frame': current_event['seg_end_frame'],
                        'start_time': seg_start_frame / self.fps,
                        'end_time': current_event['seg_end_frame'] / self.fps,
                    })
                else:
                    next_event = sorted_events[i + 1]
                    # 如果当前事件和下一个事件不连续（有间隔或重叠），则结束当前片段
                    if current_event['seg_end_frame'] < next_event['seg_start_frame']:
                        raw_segments.append({
                            'type': current_event["type"],
                            'start_frame': seg_start_frame,
                            'end_frame': current_event['seg_end_frame'],
                            'start_time': seg_start_frame / self.fps,
                            'end_time': current_event['seg_end_frame'] / self.fps,
                        })
                        seg_start_frame = None  # 重置，等待下一个事件
                    # 如果连续或重叠，继续累积到下一个事件

        # 2. 对每个原始片段进行裁剪
        final_segments = []
        for raw_seg in raw_segments:
            # 决胜时刻（victory）不裁剪无效区间
            # 因为决胜时刻的结束画面可能是无比分画面
            if raw_seg['type'] == 'victory':
                description = self.HIGHLIGHT_CONFIG[raw_seg['type']]['description']
                final_segments.append(ClipSegment(
                    start_frame=int(raw_seg['start_frame']),
                    end_frame=int(raw_seg['end_frame']),
                    start_time=raw_seg['start_time'],
                    end_time=raw_seg['end_time'],
                    clip_type=raw_seg['type'],
                    description=description,
                    parent_event=raw_seg['type']
                ))
                continue

            # 找出与该片段重叠的无效区间
            overlapping_invalid = [
                ir for ir in invalid_ranges
                if ir.start_frame <= raw_seg['end_frame'] and
                   ir.end_frame >= raw_seg['start_frame']
            ]

            if not overlapping_invalid:
                # 没有无效区间，直接添加
                description = self.HIGHLIGHT_CONFIG[raw_seg['type']]['description']
                final_segments.append(ClipSegment(
                    start_frame=int(raw_seg['start_frame']),
                    end_frame=int(raw_seg['end_frame']),
                    start_time=raw_seg['start_time'],
                    end_time=raw_seg['end_time'],
                    clip_type=raw_seg['type'],
                    description=description,
                    parent_event=raw_seg['type']
                ))
            else:
                # 裁剪无效区间
                cropped = self._crop_segment(raw_seg, overlapping_invalid)
                final_segments.extend(cropped)
        
        # 3. 过滤过短的片段
        final_segments = [
            seg for seg in final_segments
            if (seg.end_time - seg.start_time) >= self.MIN_SEGMENT_DURATION
        ]

        # 4. 如果存在胜利画面，将最后一个 victory 片段命名为"决胜时刻"
        if victory_events:
            # 从后往前找第一个 victory 类型的片段
            for i in range(len(final_segments) - 1, -1, -1):
                if final_segments[i].clip_type == 'victory':
                    final_segments[i].description = "决胜时刻"
                    break

        return final_segments

    def _crop_segment(
        self, 
        segment: dict, 
        invalid_ranges: List[InvalidRange]
    ) -> List[ClipSegment]:
        """裁剪单个片段中的无效区间"""
        
        start = segment['start_frame']
        end = segment['end_frame']
        cropped_segments = []
        
        # 按起始帧排序无效区间
        sorted_invalid = sorted(invalid_ranges, key=lambda r: r.start_frame)
        
        current_start = start
        
        for invalid in sorted_invalid:
            # 无效区间在片段内的实际边界
            invalid_start = max(invalid.start_frame, start)
            invalid_end = min(invalid.end_frame, end)
            
            # 如果无效区间之前有有效内容，生成一个子片段
            if invalid_start > current_start:
                sub_end = invalid_start - 1
                duration = (sub_end - current_start) / self.fps
                
                if duration >= self.MIN_SEGMENT_DURATION:  # 至少 3 秒
                    description = self.HIGHLIGHT_CONFIG[segment['type']]['description']
                    cropped_segments.append(ClipSegment(
                        start_frame=current_start,
                        end_frame=sub_end,
                        start_time=current_start / self.fps,
                        end_time=sub_end / self.fps,
                        clip_type=segment['type'],
                        description=f"{description} (片段{len(cropped_segments)+1})",
                        parent_event=segment['type']
                    ))

            # 更新当前起始位置到无效区间之后
            current_start = invalid_end + 1

        # 处理最后一个有效区间（最后一个无效区间之后）
        if current_start <= end:
            duration = (end - current_start) / self.fps
            if duration >= self.MIN_SEGMENT_DURATION:
                description = self.HIGHLIGHT_CONFIG[segment['type']]['description']
                cropped_segments.append(ClipSegment(
                    start_frame=current_start,
                    end_frame=end,
                    start_time=current_start / self.fps,
                    end_time=end / self.fps,
                    clip_type=segment['type'],
                    description=f"{description} (片段{len(cropped_segments)+1})",
                    parent_event=segment['type']
                ))
        
        return cropped_segments

    def add_transition(
            self,
            frame: np.ndarray,
            transition_type: str = 'fade',
            duration: int = 10
    ) -> np.ndarray:
        """
        添加转场效果
        
        Args:
            frame: 输入帧
            transition_type: 转场类型 ('fade', 'blur')
            duration: 转场持续帧数
            
        Returns:
            处理后的帧
        """
        if transition_type == 'fade':
            # 淡入淡出效果
            alpha = np.linspace(0, 1, duration)
            result = frame.copy().astype(float)
            return result.astype(np.uint8)

        elif transition_type == 'blur':
            # 模糊转场
            kernel_size = min(duration, 21)
            if kernel_size % 2 == 0:
                kernel_size += 1
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        return frame

    def extract_clip_with_transition(
            self,
            video_path: str,
            segment: ClipSegment,
            output_path: str,
            callback=None,
            add_fade: bool = True,
            clip_name: str = "",
            use_ffmpeg: bool = False,
            current_clip: int = 1,
            total_clips: int = 1
    ) -> bool:
        """
        提取单个片段，添加转场效果

        Args:
            video_path: 原视频路径
            segment: 片段信息
            output_path: 输出路径
            callback: 进度回调
            add_fade: 是否添加淡入淡出
            clip_name: 片段名称（用于回调）
            use_ffmpeg: 是否使用 FFmpeg 处理
            current_clip: 当前片段索引（从 1 开始）
            total_clips: 总片段数

        Returns:
            是否成功
        """
        if use_ffmpeg and check_ffmpeg_installed():
            return self._extract_clip_with_ffmpeg(video_path, segment, output_path, callback, add_fade, clip_name, current_clip, total_clips)
        else:
            return self._extract_clip_with_av(video_path, segment, output_path, callback, add_fade, clip_name, current_clip, total_clips)

    def _extract_clip_with_ffmpeg(
            self,
            video_path: str,
            segment: ClipSegment,
            output_path: str,
            callback=None,
            add_fade: bool = True,
            clip_name: str = "",
            current_clip: int = 1,
            total_clips: int = 1
    ) -> bool:
        """
        使用 FFmpeg 提取片段（保留音频）- 带实时进度
        
        FFmpeg 处理速度比 av 库快 10-50 倍
        """
        import re
        try:
            start_time = segment.start_time
            duration = segment.end_time - segment.start_time
            fade_duration = min(0.5, duration / 4)  # 淡入淡出时长（秒），最多 0.5 秒

            # 构建 FFmpeg 命令
            cmd = ['ffmpeg', '-y']  # -y 覆盖输出文件

            # 输入文件
            cmd.extend(['-i', video_path])

            # 设置起始时间和时长
            cmd.extend(['-ss', f'{start_time:.3f}'])
            cmd.extend(['-t', f'{duration:.3f}'])

            # 视频编码
            cmd.extend(['-c:v', 'libx264'])
            cmd.extend(['-preset', 'fast'])  # 快速编码
            cmd.extend(['-crf', '23'])  # 质量参数（0-51，越小质量越高）

            # 添加淡入淡出效果（如果启用）
            if add_fade and fade_duration > 0:
                video_filter = f'fade=t=in:st=0:d={fade_duration:.3f},fade=t=out:st={duration - fade_duration:.3f}:d={fade_duration:.3f}'
                cmd.extend(['-vf', video_filter])

            # 音频编码
            cmd.extend(['-c:a', 'aac'])
            cmd.extend(['-b:a', '192k'])

            # 输出文件（确保路径使用正斜杠）
            output_path_normalized = output_path.replace('\\', '/')
            cmd.append(output_path_normalized)

            # 执行 FFmpeg 命令，实时读取输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            # 解析 FFmpeg 进度输出
            total_seconds = duration
            last_progress_update = time.time()
            
            while True:
                # 读取一行 stderr 输出（FFmpeg 输出进度到 stderr）
                line = process.stderr.readline()
                if not line:
                    # 进程结束
                    break
                
                try:
                    line_str = line.decode('utf-8', errors='ignore').strip()
                except:
                    continue
                
                # 解析时间进度（格式：time=00:00:05.23）
                time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line_str)
                if time_match:
                    hours = int(time_match.group(1))
                    minutes = int(time_match.group(2))
                    seconds = float(time_match.group(3))
                    current_time = hours * 3600 + minutes * 60 + seconds
                    
                    # 计算进度百分比
                    if total_seconds > 0:
                        progress = min(100, int((current_time / total_seconds) * 100))
                        
                        # 限制更新频率（每 0.2 秒更新一次）
                        current_time_now = time.time()
                        if current_time_now - last_progress_update >= 0.2:
                            if callback and clip_name:
                                callback(clip_name, progress, 100, current_clip, total_clips)
                            last_progress_update = current_time_now

            # 等待进程完成
            process.wait()
            
            # 读取剩余的 stderr
            stderr_remaining = process.stderr.read()
            
            if process.returncode != 0:
                stderr_msg = stderr_remaining.decode('gbk', errors='ignore') if stderr_remaining else ''
                print(f"[FFmpeg] 错误：{stderr_msg}")
                return False

            # 完成回调
            if callback and clip_name:
                callback(clip_name, 100, 100, current_clip, total_clips)

            return True

        except Exception as e:
            print(f"[FFmpeg] 提取失败：{e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_clip_with_av(
            self,
            video_path: str,
            segment: ClipSegment,
            output_path: str,
            callback=None,
            add_fade: bool = True,
            clip_name: str = "",
            current_clip: int = 1,
            total_clips: int = 1
    ) -> bool:
        """
        使用 av 库提取片段（保留音频）- 优化版

        优化点：
        1. 使用 bgr24 格式避免 RGB↔BGR 转换
        2. 预计算淡入淡出系数，减少重复计算
        3. 使用帧索引直接 seek，避免遍历无关帧
        4. 批量编码减少 mux 调用开销
        """
        try:
            # 打开输入视频
            container = av.open(video_path)

            # 获取视频和音频流
            video_stream = None
            audio_stream = None
            for stream in container.streams:
                if stream.type == 'video':
                    video_stream = stream
                elif stream.type == 'audio':
                    audio_stream = stream

            if video_stream is None:
                container.close()
                return False

            # 计算起始帧和结束帧
            start_frame = int(segment.start_frame)
            end_frame = int(segment.end_frame)
            frame_count = end_frame - start_frame + 1
            fade_duration = min(10, frame_count // 4)

            # 计算起始时间和结束时间（使用视频帧率）
            fps = float(video_stream.average_rate)
            start_time = start_frame / fps
            end_time = (end_frame + 1) / fps

            # 创建输出容器
            output_path = output_path.replace('\\', '/')
            output_container = av.open(output_path, 'w')
            output_video_stream = output_container.add_stream('h264', rate=video_stream.average_rate)
            output_video_stream.width = video_stream.width
            output_video_stream.height = video_stream.height
            output_video_stream.pix_fmt = 'yuv420p'

            # 如果有音频流，添加音频
            output_audio_stream = None
            if audio_stream:
                output_audio_stream = output_container.add_stream('aac')
                output_audio_stream.layout = audio_stream.layout
                output_audio_stream.sample_rate = audio_stream.sample_rate

            processed_frames = 0

            # 直接 seek 到起始位置，避免遍历无关帧
            container.seek(0)
            
            # 预计算淡入淡出系数（避免每帧重复计算）
            fade_in_alphas = None
            fade_out_alphas = None
            if add_fade and fade_duration > 0:
                fade_in_alphas = np.linspace(0, 1, fade_duration, dtype=np.float32)
                fade_out_alphas = np.linspace(1, 0, fade_duration, dtype=np.float32)

            # 使用生成器逐帧处理，减少内存占用
            frame_iterator = container.decode(video_stream)
            
            # 跳过起始帧之前的帧
            for frame in frame_iterator:
                if frame.time >= start_time:
                    break
            
            # 预分配黑帧（用于淡入淡出）
            black_frame = None

            # 编码视频 - 逐帧处理
            while True:
                if frame.time >= end_time:
                    break
                
                # 转换为 BGR 格式（避免后续转换）
                frame_bgr = frame.to_ndarray(format='bgr24')
                
                # 添加淡入淡出效果（优化版）
                if add_fade:
                    if processed_frames < fade_duration:
                        # 淡入
                        if black_frame is None:
                            black_frame = np.zeros_like(frame_bgr)
                        alpha = fade_in_alphas[processed_frames]
                        frame_bgr = (frame_bgr * alpha + black_frame * (1 - alpha)).astype(np.uint8)
                    elif processed_frames >= frame_count - fade_duration:
                        # 淡出
                        if black_frame is None:
                            black_frame = np.zeros_like(frame_bgr)
                        fade_out_idx = processed_frames - (frame_count - fade_duration)
                        alpha = fade_out_alphas[fade_out_idx]
                        frame_bgr = (frame_bgr * alpha + black_frame * (1 - alpha)).astype(np.uint8)

                # 直接使用 BGR 格式创建帧
                output_frame = av.VideoFrame.from_ndarray(frame_bgr, format='bgr24')
                output_container.mux(output_video_stream.encode(output_frame))

                processed_frames += 1
                if processed_frames >= frame_count:
                    break
                    
                # 获取下一帧
                try:
                    frame = next(frame_iterator)
                except StopIteration:
                    break

                if callback and processed_frames % 5 == 0:  # 降低回调频率
                    callback(clip_name, processed_frames, frame_count, current_clip, total_clips)

            # 处理音频（如果有）
            if audio_stream and output_audio_stream:
                container.seek(int(start_time * av.time_base), stream=audio_stream)
                next_audio_pts = 0
                
                for packet in container.demux(audio_stream):
                    for audio_frame in packet.decode():
                        audio_frame.pts = next_audio_pts
                        next_audio_pts += audio_frame.samples
                        output_container.mux(output_audio_stream.encode(audio_frame))
                        
                        # 检查是否已超出结束时间
                        if audio_frame.time >= end_time:
                            break

            # 完成编码
            output_container.mux(output_video_stream.encode())
            if output_audio_stream:
                output_container.mux(output_audio_stream.encode())

            container.close()
            output_container.close()
            return True

        except Exception as e:
            print(f"[ERROR] av 库提取失败：{e}")
            import traceback
            traceback.print_exc()
            raise

    def extract_all_clips(
            self,
            video_path: str,
            segments: List[ClipSegment],
            output_dir: str,
            callback=None,
            add_fade: bool = True,
            extract_first_only: bool = False,  # 默认提取所有片段
            use_ffmpeg: bool = False  # 是否使用 FFmpeg 处理
    ) -> List[str]:
        """提取所有片段"""
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        # 如果只提取第一段，则只处理第一个 segment
        segments_to_process = segments[:1] if extract_first_only else segments

        print(f"\n✂️ 开始提取高光片段，共 {len(segments_to_process)} 个:")

        for i, segment in enumerate(segments_to_process):
            filename = f"clip_{i + 1:03d}_{segment.clip_type}_{segment.start_time:.1f}s.mp4"
            output_path = os.path.join(output_dir, filename)

            # 最后一个片段不添加渐隐效果
            is_last = (i == len(segments_to_process) - 1)
            fade_for_clip = add_fade and not is_last

            if self.extract_clip_with_transition(
                    video_path, segment, output_path, callback, fade_for_clip, filename, use_ffmpeg,
                    i + 1, len(segments_to_process)  # 传递当前片段索引和总数
            ):
                output_paths.append(output_path)
                print(f"   ✅ [{i + 1}/{len(segments_to_process)}] {segment.description}")
                print(f"      保存路径：{output_path}")
                print(f"      时间：{segment.start_time:.1f}s ~ {segment.end_time:.1f}s")

        return output_paths

    def merge_clips(
            self,
            clip_paths: List[str],
            output_path: str,
            transition_frames: int = 10,
            callback=None,
            use_ffmpeg: bool = False
    ) -> bool:
        """
        拼接多个片段为一个完整视频

        Args:
            clip_paths: 片段文件路径列表
            output_path: 输出路径
            transition_frames: 转场帧数
            callback: 进度回调
            use_ffmpeg: 是否使用 FFmpeg 处理

        Returns:
            是否成功
        """
        if use_ffmpeg and check_ffmpeg_installed():
            return self._merge_clips_with_ffmpeg(clip_paths, output_path, callback)
        else:
            return self._merge_clips_with_av(clip_paths, output_path, transition_frames, callback)

    def _merge_clips_with_ffmpeg(
            self,
            clip_paths: List[str],
            output_path: str,
            callback=None
    ) -> bool:
        """
        使用 FFmpeg 拼接片段（保留音频）- 带实时进度

        使用 FFmpeg 的 concat demuxer 进行快速拼接
        """
        import re
        try:
            if not clip_paths:
                return False

            # 创建临时文件列表
            import tempfile
            temp_dir = tempfile.gettempdir()
            list_file = os.path.join(temp_dir, f"ffmpeg_concat_{os.getpid()}.txt")

            # 写入文件列表（FFmpeg concat demuxer 格式）
            with open(list_file, 'w', encoding='utf-8') as f:
                for clip_path in clip_paths:
                    # 路径需要转义单引号
                    escaped_path = clip_path.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # 构建 FFmpeg 命令
            cmd = ['ffmpeg', '-y']  # -y 覆盖输出文件

            # 使用 concat demuxer
            cmd.extend(['-f', 'concat'])
            cmd.extend(['-safe', '0'])  # 允许使用文件路径
            cmd.extend(['-i', list_file])  # 输入文件列表

            # 视频编码（重新编码以保持一致性）
            cmd.extend(['-c:v', 'libx264'])
            cmd.extend(['-preset', 'fast'])
            cmd.extend(['-crf', '23'])

            # 音频编码
            cmd.extend(['-c:a', 'aac'])
            cmd.extend(['-b:a', '192k'])

            # 输出文件
            output_path_normalized = output_path.replace('\\', '/')
            cmd.append(output_path_normalized)

            # 执行 FFmpeg 命令，实时读取输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            # 解析 FFmpeg 进度输出
            last_progress_update = time.time()
            last_frame_count = 0
            
            while True:
                # 读取一行 stderr 输出（FFmpeg 输出进度到 stderr）
                line = process.stderr.readline()
                if not line:
                    # 进程结束
                    break
                
                try:
                    line_str = line.decode('utf-8', errors='ignore').strip()
                except:
                    continue
                
                # 解析帧数进度（格式：frame=  123）
                frame_match = re.search(r'frame=\s*(\d+)', line_str)
                if frame_match:
                    current_frame = int(frame_match.group(1))
                    
                    # 限制更新频率（每 0.2 秒更新一次）
                    current_time_now = time.time()
                    if current_time_now - last_progress_update >= 0.2:
                        # 由于不知道总帧数，使用帧数增长来估算进度
                        if current_frame > last_frame_count:
                            last_frame_count = current_frame
                        # 使用帧数作为进度指示（百分比通过时间估算）
                        if callback:
                            callback("拼接中", int(current_time_now % 100), 100, 1, 1)
                        last_progress_update = current_time_now

            # 等待进程完成
            process.wait()
            
            # 读取剩余的 stderr
            stderr_remaining = process.stderr.read()

            # 清理临时文件
            try:
                os.remove(list_file)
            except Exception:
                pass

            if process.returncode != 0:
                stderr_msg = stderr_remaining.decode('gbk', errors='ignore') if stderr_remaining else ''
                print(f"[FFmpeg] 合并错误：{stderr_msg}")
                return False

            if callback:
                callback("合并完成", 100, 100, 1, 1)

            return True

        except Exception as e:
            print(f"[FFmpeg] 合并失败：{e}")
            import traceback
            traceback.print_exc()
            return False

    def _merge_clips_with_av(
            self,
            clip_paths: List[str],
            output_path: str,
            transition_frames: int = 10,
            callback=None
    ) -> bool:
        """使用 av 库拼接片段（保留音频）"""
        try:
            if not clip_paths:
                return False

            # 打开所有输入容器
            containers = []
            video_streams = []
            audio_streams = []

            for clip_path in clip_paths:
                container = av.open(clip_path)
                containers.append(container)

                video_stream = None
                audio_stream = None
                for stream in container.streams:
                    if stream.type == 'video':
                        video_stream = stream
                    elif stream.type == 'audio':
                        audio_stream = stream

                if video_stream is None:
                    for c in containers:
                        c.close()
                    return False

                video_streams.append(video_stream)
                audio_streams.append(audio_stream)

            # 使用第一个视频的流参数
            first_video_stream = video_streams[0]

            # 创建输出容器（确保路径使用正斜杠）
            output_path = output_path.replace('\\', '/')
            output_container = av.open(output_path, 'w')
            output_video_stream = output_container.add_stream('h264', rate=first_video_stream.average_rate)
            output_video_stream.width = first_video_stream.width
            output_video_stream.height = first_video_stream.height
            output_video_stream.pix_fmt = 'yuv420p'

            # 添加音频流（如果第一个片段有音频）
            output_audio_stream = None
            if audio_streams[0]:
                output_audio_stream = output_container.add_stream('aac')
                output_audio_stream.layout = audio_streams[0].layout
                output_audio_stream.sample_rate = audio_streams[0].sample_rate

            # 统计总帧数用于进度
            total_frames = 0
            for container in containers:
                for stream in container.streams:
                    if stream.type == 'video':
                        total_frames += stream.frames
                        break

            processed_frames = 0
            total_clips = len(clip_paths)

            # 音频时间戳跟踪（用于保持连续性）
            # 使用秒数跟踪，最后转换为输出流的 PTS 单位
            next_audio_pts_sec = 0.0

            for clip_idx, (container, video_stream, audio_stream) in enumerate(zip(containers, video_streams, audio_streams)):
                is_last = clip_idx == total_clips - 1
                # 获取当前片段的帧数
                clip_frame_count = 0
                for stream in container.streams:
                    if stream.type == 'video':
                        clip_frame_count = stream.frames
                        break
                clip_frames_written = 0

                # 处理视频帧
                for frame in container.decode(video_stream):
                    # 转换为 numpy 数组
                    frame_array = frame.to_ndarray(format='rgb24')
                    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

                    # 淡入效果（每个片段开头，除了第一个）
                    if clip_frame_count > 0 and clip_frames_written < transition_frames and clip_idx > 0:
                        alpha = clip_frames_written / transition_frames
                        frame_bgr = cv2.addWeighted(
                            np.zeros_like(frame_bgr), 1 - alpha, frame_bgr, alpha, 0
                        )

                    # 淡出效果（每个片段结尾，除了最后一个）
                    if not is_last and clip_frames_written >= clip_frame_count - transition_frames:
                        alpha = (clip_frame_count - clip_frames_written) / transition_frames
                        frame_bgr = cv2.addWeighted(
                            np.zeros_like(frame_bgr), 1 - alpha, frame_bgr, alpha, 0
                        )

                    # 转换回 RGB 并写入
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    output_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
                    output_container.mux(output_video_stream.encode(output_frame))

                    clip_frames_written += 1
                    processed_frames += 1

                    # 每 10 帧更新一次进度
                    if callback and processed_frames % 10 == 0:
                        callback(f"拼接中：{processed_frames}/{total_frames}", processed_frames, total_frames)

                # 处理音频 - 重新计算时间戳保持连续性
                if audio_stream and output_audio_stream:
                    container.seek(0)
                    for packet in container.demux(audio_stream):
                        for audio_frame in packet.decode():
                            # 使用输出流的时间基准计算 PTS
                            # 输出流的 time_base 通常是 1/sample_rate
                            audio_frame.pts = int(next_audio_pts_sec * output_audio_stream.sample_rate)

                            # 更新下一个 PTS（当前时间 + 当前帧时长）
                            next_audio_pts_sec += audio_frame.samples / output_audio_stream.sample_rate

                            # 编码音频帧并处理可能的 None 返回值
                            encoded_packets = output_audio_stream.encode(audio_frame)
                            if encoded_packets:
                                for pkt in encoded_packets:
                                    output_container.mux(pkt)

                # 每个片段完成后更新进度
                if callback:
                    callback(f"拼接中：{clip_idx + 1}/{total_clips}", processed_frames, total_frames)

            # 完成编码 - 刷新缓冲区
            # 刷新视频缓冲区
            for pkt in output_video_stream.encode(None):
                if pkt:
                    output_container.mux(pkt)
            
            # 刷新音频缓冲区
            if output_audio_stream:
                for pkt in output_audio_stream.encode(None):
                    if pkt:
                        output_container.mux(pkt)

            # 关闭所有容器
            for c in containers:
                c.close()
            output_container.close()

            return True

        except Exception as e:
            print(f"[ERROR] av 库拼接失败：{e}")
            raise

    def extract_and_merge(
            self,
            video_path: str,
            segments: List[ClipSegment],
            output_dir: str,
            merge_output: str,
            callback=None
    ) -> Tuple[List[str], str]:
        """
        提取所有片段并拼接成一个完整视频
        
        Args:
            video_path: 原视频路径
            segments: 片段列表
            output_dir: 单独片段输出目录
            merge_output: 拼接后视频路径
            callback: 进度回调
            
        Returns:
            (单独片段路径列表，拼接视频路径)
        """
        # 先提取所有片段
        clip_paths = self.extract_all_clips(video_path, segments, output_dir, callback)

        # 拼接片段
        if clip_paths:
            self.merge_clips(clip_paths, merge_output, callback)

        return clip_paths, merge_output

    def get_video_info(self, video_path: str) -> Dict:
        """获取视频基本信息"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}

        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return info
