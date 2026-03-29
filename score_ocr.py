"""
Dota2 比分区域人头数识别模块
使用 EasyOCR 识别比分区域的人头数变化
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

# 检测间隔帧数
DETECT_INTERVAL_FRAMES = 30
# 时间置信度阈值
TIME_CONFIDENCE_THRESHOLD = 0.9

# 尝试导入 easyocr
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("❌ easyocr 未安装，请安装：pip install easyocr")


@dataclass
class ScoreInfo:
    """比分信息"""
    radiant_kills: int  # 天灾人头数
    dire_kills: int     # 夜魇人头数
    frame_idx: int
    timestamp: float
    time_str: str = "??:??"  # 比赛时间
    time_conf: float = 0.0   # 时间置信度
    old_score: Optional[Tuple[int, int]] = None  # (old_radiant, old_dire)
    new_score: Optional[Tuple[int, int]] = None  # (new_radiant, new_dire)


@dataclass
class KillEvent:
    """人头变化事件"""
    frame: int
    time: float
    team: str  # 'radiant' or 'dire'
    old_score: int
    new_score: int
    confidence: float


class ScoreOCRDetector:
    """比分区域 OCR 检测器 - 使用 EasyOCR"""

    def __init__(self, use_ocr: bool = True, device: str = 'auto'):
        """
        初始化检测器

        Args:
            use_ocr: 是否使用 EasyOCR
            device: 推理设备 ('cuda', 'cpu', 'auto')
        """
        self.use_ocr = use_ocr and EASYOCR_AVAILABLE
        self.device = device
        self.reader = None

        if self.use_ocr:
            try:
                # 自动选择设备
                if self.device == 'auto':
                    import torch
                    use_gpu = torch.cuda.is_available()
                else:
                    use_gpu = self.device == 'cuda'

                # 初始化 EasyOCR（只识别英文数字）
                self.reader = easyocr.Reader(
                    ['en'],
                    gpu=use_gpu,
                    verbose=False
                )
                if use_gpu:
                    pass  # print("✅ EasyOCR 已加载到 GPU")
                else:
                    pass  # print("ℹ️ EasyOCR 使用 CPU 模式")
            except Exception as e:
                # print(f"⚠️  EasyOCR 初始化失败：{e}")
                self.use_ocr = False

        # 人头数变化历史
        self.score_history: List[ScoreInfo] = []
        self.kill_events: List[KillEvent] = []

        # 比分区域 ROI
        self.score_area_roi = None
        
        # 人头数缓存（初始为 0:0）
        self.last_radiant = 0
        self.last_dire = 0

        # 时间缓存和校验
        self.last_time_str = "0:00"
        self.last_time_conf = 0.0
        self.time_history: List[Tuple[int, str, float]] = []  # (frame_idx, time_str, conf)
        self.time_trend = None  # None, "increasing", "decreasing"

        # 初始化标志（用于处理视频不是从 0:0 开始的情况）
        self.initialized = False  # 是否已完成初始值校准
        self.init_skip_frames = 3  # 前 3 次检测用于校准，不触发人头事件

        # 验证模式状态跟踪（用于过滤误识别）
        self.verifying = False  # 是否处于验证模式
        self.verify_target = None  # 间隔帧识别到的人头数 (radiant, dire)
        self.verify_frames = []  # 验证帧的识别结果 [(radiant, dire, frame_idx), ...]
        self.verify_max_frames = 3  # 最多验证 3 帧
        self.verify_kill_detected = False  # 间隔帧是否检测到人头变化

    def set_score_area(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None):
        """
        设置比分区域 ROI

        Args:
            frame: 视频帧（用于自动检测）
            roi: 手动指定的 ROI (x, y, w, h)，如果为 None 则尝试自动检测
        """
        if roi:
            self.score_area_roi = roi
        else:
            # 自动检测比分区域（基于常见 Dota2 UI 布局）
            h, w = frame.shape[:2]
            self.score_area_roi = (
                int(w * 0.35),   # x
                int(h * 0.02),   # y
                int(w * 0.30),   # width
                int(h * 0.08)    # height
            )

        # print(f"📊 比分区域 ROI: {self.score_area_roi}")

    def extract_score_area(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """从帧中提取比分区域"""
        if self.score_area_roi is None:
            self.set_score_area(frame)

        x, y, w, h = self.score_area_roi
        h_frame, w_frame = frame.shape[:2]

        # 边界检查
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)

        return frame[y:y+h, x:x+w]

    def recognize_score(self, score_image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        识别比分图片中的人头数

        Args:
            score_image: 比分区域图片

        Returns:
            (radiant_kills, dire_kills) 或 None
        """
        if self.use_ocr and self.reader:
            return self._easyocr_recognize(score_image)
        else:
            pass  # print("⚠️  OCR 未启用，无法识别人头数")
            return None

    def _recognize_time_in_region(self, region: np.ndarray) -> Tuple[Optional[str], float]:
        """
        识别中间区域的时间（格式如 0:00）
        返回：(时间字符串，置信度)
        """
        try:
            # 使用允许列表识别数字和冒号
            results = self.reader.readtext(region, paragraph=False, allowlist='0123456789:')

            if not results:
                return None, 0.0

            # 提取识别到的文本
            texts = []
            for (bbox, text, conf) in results:
                if text and ":" in text:
                    cleaned = ''.join(c for c in text if c.isdigit() or c == ':')
                    if cleaned:
                        texts.append({'text': cleaned, 'confidence': conf})
                if text and ":" not in text and len(text) >= 3:
                    index = len(text) - 2
                    cleaned = text[:index] + ":" + text[index:]
                    if cleaned:
                        texts.append({'text': cleaned, 'confidence': conf})

            if not texts:
                return None, 0.0

            best_text = max(texts, key=lambda t: t['confidence'])
            
            # 格式化时间：将 27:0 格式化为 27:00
            time_str = best_text['text']
            formatted_time = self._format_time_str(time_str)
            
            return formatted_time, best_text['confidence']

        except Exception as e:
            return None, 0.0
    
    def _format_time_str(self, time_str: str) -> str:
        """
        格式化时间字符串，将 27:0 格式化为 27:00
        """
        if not time_str or ':' not in time_str:
            return time_str
        
        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = parts[0]
                seconds = parts[1]
                # 秒数部分补齐到 2 位
                if len(seconds) == 1:
                    seconds = seconds + '0'
                elif len(seconds) == 0:
                    seconds = '00'
                return f"{minutes}:{seconds}"
        except:
            pass
        return time_str

    def _parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        """将时间字符串转换为秒数"""
        if not time_str or time_str == "??:??":
            return None
        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
        except:
            pass
        return None

    def _seconds_to_time_str(self, seconds: float) -> str:
        """将秒数转换为时间字符串（格式：1:03 而不是 01:03）"""
        if seconds is None:
            return "??:??"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def _detect_time_trend(self) -> Optional[str]:
        """根据历史数据检测时间变化趋势"""
        if len(self.time_history) < 2:
            return self.time_trend

        recent = [h for h in self.time_history[-5:] if h[2] >= TIME_CONFIDENCE_THRESHOLD]
        if len(recent) < 2:
            return self.time_trend

        first_sec = self._parse_time_to_seconds(recent[0][1])
        last_sec = self._parse_time_to_seconds(recent[-1][1])

        if first_sec is not None and last_sec is not None:
            if last_sec > first_sec:
                return "increasing"
            elif last_sec < first_sec:
                return "decreasing"
        return self.time_trend

    def _get_expected_time_range(self, fps: float) -> Optional[Tuple[float, float]]:
        """
        根据历史数据和趋势，返回预期的时间范围
        返回：(min_sec, max_sec) 或 None
        """
        if not self.time_history:
            return None

        recent = [h for h in self.time_history[-5:] if h[2] >= TIME_CONFIDENCE_THRESHOLD]
        if not recent:
            return None

        last_frame, last_time_str, _ = recent[-1]
        last_sec = self._parse_time_to_seconds(last_time_str)

        if last_sec is None:
            return None

        # 计算帧数差（假设当前帧是最近检测帧 + DETECT_INTERVAL_FRAMES）
        frames_elapsed = DETECT_INTERVAL_FRAMES
        max_change = frames_elapsed / fps

        if self.time_trend == "increasing":
            return (last_sec, last_sec + max_change * 1.5)
        elif self.time_trend == "decreasing":
            return (last_sec - max_change * 1.5, last_sec)
        else:
            return (last_sec - max_change, last_sec + max_change)

    def _is_time_reasonable(self, time_str: str, expected_range: Optional[Tuple[float, float]]) -> bool:
        """检查时间是否在预期范围内"""
        if expected_range is None:
            return True

        t_sec = self._parse_time_to_seconds(time_str)
        if t_sec is None:
            return False

        min_sec, max_sec = expected_range
        return (min_sec - 3) <= t_sec <= (max_sec + 3)

    def _easyocr_recognize(self, score_image: np.ndarray) -> Optional[Tuple[int, int, str, float]]:
        """
        使用 EasyOCR 识别比分和时间
        返回：(radiant_kills, dire_kills, time_str, time_conf)
        """
        try:
            # 放大 4 倍以提高识别准确率
            score_image_scaled = cv2.resize(score_image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            img_height, img_width = score_image_scaled.shape[:2]

            # 计算区域边界（左侧 25%，中间 50%，右侧 25%）
            left_region_width = int(img_width * 0.25)
            middle_region_start = left_region_width
            middle_region_end = int(img_width * 0.75)
            right_region_start = middle_region_end

            # 提取三个区域
            left_region = score_image_scaled[:, :left_region_width]
            middle_region = score_image_scaled[:, middle_region_start:middle_region_end]
            right_region = score_image_scaled[:, right_region_start:]

            # 识别左侧人头数
            left_results = self.reader.readtext(left_region, paragraph=False, allowlist='0123456789')
            left_digits = []
            for (bbox, text, conf) in left_results:
                if text and text.isdigit():
                    left_digits.append({'text': int(text), 'confidence': conf})
            left_kills = max(left_digits, key=lambda d: d['confidence'])['text'] if left_digits else None

            # 识别右侧人头数
            right_results = self.reader.readtext(right_region, paragraph=False, allowlist='0123456789')
            right_digits = []
            for (bbox, text, conf) in right_results:
                if text and text.isdigit():
                    right_digits.append({'text': int(text), 'confidence': conf})
            right_kills = max(right_digits, key=lambda d: d['confidence'])['text'] if right_digits else None

            # 识别中间时间
            time_str, time_conf = self._recognize_time_in_region(middle_region)

            return left_kills, right_kills, time_str, time_conf

        except Exception as e:
            # print(f"EasyOCR 识别失败：{e}")
            return None, None, None, 0.0

    def analyze_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> Optional[ScoreInfo]:
        """分析单帧，使用固定 ROI 提取比分区域并识别人头数和时间"""
        # 提取比分区域
        score_area = self.extract_score_area(frame)
        if score_area is None:
            return None

        return self._process_score_area(score_area, frame_idx, fps)

    def analyze_frame_with_score_area(self, score_area: np.ndarray, frame_idx: int, fps: float) -> Optional[ScoreInfo]:
        """
        分析单帧，使用给定的比分区域（YOLO 检测）进行识别
        
        Args:
            score_area: YOLO 检测到的比分区域
            frame_idx: 帧索引
            fps: 视频帧率
            
        Returns:
            ScoreInfo 或 None
        """
        return self._process_score_area(score_area, frame_idx, fps)

    def _process_score_area(self, score_area: np.ndarray, frame_idx: int, fps: float) -> Optional[ScoreInfo]:
        """处理比分区域，识别人头数和时间"""
        # 识别人头数和时间
        radiant, dire, time_str, time_conf = self._easyocr_recognize(score_area)

        # 初始值校准逻辑：处理视频不是从 0:0 开始的情况
        if not self.initialized:
            if radiant is not None and dire is not None:
                # 检查比赛时间，如果时间 > 30 秒，说明视频从比赛中段开始
                time_sec = self._parse_time_to_seconds(time_str) if time_str else 0
                if time_sec is None:
                    time_sec = 0
                
                # 如果时间 > 30 秒 或 人头数 > 0，视为非 0:0 开局
                if time_sec > 30 or radiant > 0 or dire > 0:
                    print(f"[DEBUG] 初始值校准：视频从比赛中段开始，设置初始人头数 {radiant}:{dire} (比赛时间：{time_str})")
                    self.last_radiant = radiant
                    self.last_dire = dire
                    self.initialized = True
                    # 直接返回，不触发人头事件
                    # 继续后续逻辑，但不会检测人头变化
                else:
                    # 时间接近 0:00，可能是游戏刚开始，需要继续观察
                    # 前 init_skip_frames 次检测用于校准，不触发人头事件
                    if hasattr(self, '_init_count'):
                        self._init_count += 1
                    else:
                        self._init_count = 1
                    
                    if self._init_count >= self.init_skip_frames:
                        # 校准完成，采用当前识别值作为初始值
                        print(f"[DEBUG] 初始值校准：完成 {self.init_skip_frames} 次检测，设置初始人头数 {radiant}:{dire}")
                        self.last_radiant = radiant
                        self.last_dire = dire
                        self.initialized = True
                    # 继续后续逻辑，但不会检测人头变化

        # 验证模式处理
        if self.verifying:
            # 收集验证帧结果
            if radiant is not None and dire is not None:
                self.verify_frames.append((radiant, dire, frame_idx))
            
            # 检查是否收集到足够的验证帧
            if len(self.verify_frames) >= self.verify_max_frames:
                # 验证完成，进行判定（会更新 self.last_radiant/self.last_dire）
                self._finalize_verification()
                # 验证完成后，使用验证后的值
                radiant = self.last_radiant
                dire = self.last_dire
            else:
                # 验证进行中，使用上次缓存的值（保持不变）
                radiant = self.last_radiant
                dire = self.last_dire
        else:
            # 正常模式：进行人头数校验
            
            # 人头数校验逻辑 1：人头数只能增加，不能减少
            # 如果识别到的值比上次小，说明是误识别，使用上次结果
            if radiant is not None and self.last_radiant is not None:
                if radiant < self.last_radiant:
                    print(f"[DEBUG] 人头数校正：天灾 {self.last_radiant}→{radiant} (减少)，使用上次值 {self.last_radiant}")
                    radiant = self.last_radiant
            if dire is not None and self.last_dire is not None:
                if dire < self.last_dire:
                    print(f"[DEBUG] 人头数校正：夜魇 {self.last_dire}→{dire} (减少)，使用上次值 {self.last_dire}")
                    dire = self.last_dire

            # 人头数校验逻辑 2：单次变化不能超过阈值（最多 5 个人头）
            # 如果变化太大，说明是误识别，使用上次结果
            MAX_KILL_CHANGE = 5  # 单次最多变化 5 个人头
            if radiant is not None and self.last_radiant is not None:
                if radiant - self.last_radiant > MAX_KILL_CHANGE:
                    print(f"[DEBUG] 人头数校正：天灾 {self.last_radiant}→{radiant} (+{radiant-self.last_radiant} 变化过大)，使用上次值 {self.last_radiant}")
                    radiant = self.last_radiant
            if dire is not None and self.last_dire is not None:
                if dire - self.last_dire > MAX_KILL_CHANGE:
                    print(f"[DEBUG] 人头数校正：夜魇 {self.last_dire}→{dire} (+{dire-self.last_dire} 变化过大)，使用上次值 {self.last_dire}")
                    dire = self.last_dire

            # 如果没有识别到人头数，使用上次结果
            if radiant is None:
                radiant = self.last_radiant
            if dire is None:
                dire = self.last_dire

            # 检测是否有人头数变化（用于触发验证）
            # 注意：未初始化时不触发人头事件
            has_kill_change = False
            if self.initialized:
                has_kill_change = (radiant > self.last_radiant) or (dire > self.last_dire)

            # 如果检测到人头变化，触发验证模式
            if has_kill_change:
                # 触发验证，但不立即更新缓存
                self.verifying = True
                self.verify_target = (radiant, dire)
                self.verify_frames = []
                self.verify_kill_detected = True
                print(f"[DEBUG] 检测到人头变化：{radiant}:{dire}，启动验证模式")
                # 验证期间保持原值不变，等待验证完成后再更新
                # 当前帧的 radiant/dire 也使用原值，等待验证完成后再更新
                radiant = self.last_radiant
                dire = self.last_dire
            else:
                # 没有人头变化，正常更新缓存
                self.last_radiant = radiant
                self.last_dire = dire

        # 时间处理逻辑
        time_display = "??:??"
        
        # 检测时间变化趋势
        self.time_trend = self._detect_time_trend()
        
        # 获取预期时间范围
        expected_range = self._get_expected_time_range(fps)
        
        # 情况 1: 识别失败或置信度低，使用预期时间或上次结果
        if time_str is None or time_conf < TIME_CONFIDENCE_THRESHOLD:
            if expected_range is not None:
                min_sec, max_sec = expected_range
                expected_sec = (min_sec + max_sec) / 2
                time_display = self._seconds_to_time_str(expected_sec)
            else:
                time_display = self.last_time_str
        # 情况 2: 置信度高 (>=0.9) 直接使用识别到的时间（不再判断范围）
        elif time_conf >= 0.9:
            time_display = time_str
            self.last_time_str = time_str
            self.last_time_conf = time_conf
            self.time_history.append((frame_idx, time_str, time_conf))
        # 情况 3: 置信度中等 (0.7-0.9)，检查是否超出预期范围
        elif not self._is_time_reasonable(time_str, expected_range):
            time_display = self.last_time_str
            # 时间修正日志（调试用）
            # print(f"  [时间修正] 帧 {frame_idx}: 识别到 '{time_str}' ({time_conf:.2f}), "
            #       f"超出预期范围，使用上次值 '{self.last_time_str}' (趋势：{self.time_trend})")
        # 情况 4: 置信度中等且时间合理，使用识别到的时间
        else:
            time_display = time_str
            self.last_time_str = time_str
            self.last_time_conf = time_conf
            self.time_history.append((frame_idx, time_str, time_conf))

        timestamp = frame_idx / fps

        # 获取旧分数
        old_radiant, old_dire = None, None
        if self.score_history:
            prev_score = self.score_history[-1]
            old_radiant, old_dire = prev_score.radiant_kills, prev_score.dire_kills

        score_info = ScoreInfo(
            radiant_kills=radiant,
            dire_kills=dire,
            frame_idx=frame_idx,
            timestamp=timestamp,
            time_str=time_display,
            time_conf=time_conf,
            old_score=(old_radiant, old_dire) if old_radiant is not None else None,
            new_score=(radiant, dire)
        )

        # 检测人头变化
        self._detect_kill_event(score_info)

        return score_info

    def _finalize_verification(self):
        """
        验证完成后的判定逻辑
        
        验证逻辑：
        1. 统计验证帧中出现次数最多的值（多数投票）
        2. 与间隔帧的原始值比较：
           - 一致 → 确认变化有效
           - 验证值 > 间隔帧值 → 人头继续增加，采用验证值
           - 验证值 < 间隔帧值 → 间隔帧是误识别，采用验证值并回溯修正
        """
        if not self.verify_frames:
            # 没有有效验证帧，保持原值
            self.verifying = False
            self.verify_target = None
            self.verify_frames = []
            print(f"[DEBUG] 验证完成：无有效验证帧，保持原值 {self.last_radiant}:{self.last_dire}")
            return
        
        # 多数投票：统计验证帧中出现次数最多的值
        radiant_counts = Counter([f[0] for f in self.verify_frames])
        dire_counts = Counter([f[1] for f in self.verify_frames])
        
        validated_radiant = radiant_counts.most_common(1)[0][0]
        validated_dire = dire_counts.most_common(1)[0][0]
        
        target_radiant, target_dire = self.verify_target
        
        # 判定逻辑
        if validated_radiant == target_radiant and validated_dire == target_dire:
            # 验证一致，确认变化有效
            print(f"[DEBUG] 验证完成：验证一致 {validated_radiant}:{validated_dire}，确认变化有效")
            # 更新缓存为验证后的值
            self.last_radiant = validated_radiant
            self.last_dire = validated_dire
        elif validated_radiant > target_radiant or validated_dire > target_dire:
            # 验证值更大，说明人头继续增加，采用验证值
            print(f"[DEBUG] 验证完成：验证值更大 {target_radiant}:{target_dire}→{validated_radiant}:{validated_dire}，采用验证值")
            self.last_radiant = validated_radiant
            self.last_dire = validated_dire
        elif validated_radiant < target_radiant or validated_dire < target_dire:
            # 验证值更小，说明间隔帧是误识别，采用验证值并回溯修正
            print(f"[DEBUG] 验证完成：间隔帧误识别 {target_radiant}:{target_dire}→{validated_radiant}:{validated_dire}，修正历史")
            self.last_radiant = validated_radiant
            self.last_dire = validated_dire

            # 回溯修正 score_history 中的错误值
            if self.score_history:
                for score in self.score_history:
                    if score.radiant_kills == target_radiant and score.dire_kills == target_dire:
                        score.radiant_kills = validated_radiant
                        score.dire_kills = validated_dire
                        score.new_score = (validated_radiant, validated_dire)
        
        # 退出验证模式
        self.verifying = False
        self.verify_target = None
        self.verify_frames = []
        self.verify_kill_detected = False

    def _detect_kill_event(self, current_score: ScoreInfo):
        """检测人头数变化事件"""
        if not self.score_history:
            self.score_history.append(current_score)
            return

        prev_score = self.score_history[-1]

        # 检测天灾人头变化
        if current_score.radiant_kills > prev_score.radiant_kills:
            event = KillEvent(
                frame=current_score.frame_idx,
                time=current_score.timestamp,
                team='radiant',
                old_score=prev_score.radiant_kills,
                new_score=current_score.radiant_kills,
                confidence=0.9
            )
            self.kill_events.append(event)

        # 检测夜魇人头变化
        if current_score.dire_kills > prev_score.dire_kills:
            event = KillEvent(
                frame=current_score.frame_idx,
                time=current_score.timestamp,
                team='dire',
                old_score=prev_score.dire_kills,
                new_score=current_score.dire_kills,
                confidence=0.9
            )
            self.kill_events.append(event)

        self.score_history.append(current_score)

    def get_kill_events(self, time_window: float = 2.0) -> List[KillEvent]:
        """获取人头变化事件（合并时间窗口内的重复事件）"""
        if not self.kill_events:
            return []

        events = sorted(self.kill_events, key=lambda e: e.time)
        merged = []

        current_group = [events[0]]

        for event in events[1:]:
            if event.time - current_group[-1].time <= time_window:
                current_group.append(event)
            else:
                best_event = max(current_group, key=lambda e: e.new_score - e.old_score)
                merged.append(best_event)
                current_group = [event]

        if current_group:
            best_event = max(current_group, key=lambda e: e.new_score - e.old_score)
            merged.append(best_event)

        return merged

    def reset(self):
        """重置历史数据"""
        self.score_history.clear()
        self.kill_events.clear()
        # 重置缓存
        self.last_radiant = 0
        self.last_dire = 0
        self.last_time_str = "0:00"
        self.last_time_conf = 0.0
        self.time_history.clear()
        self.time_trend = None
        # 重置验证状态
        self.verifying = False
        self.verify_target = None
        self.verify_frames = []
        self.verify_kill_detected = False
        # 重置初始化状态
        self.initialized = False
        self.init_skip_frames = 3
        self._init_count = 0


class KillHighlightDetector:
    """人头变化高光检测器"""

    def __init__(self, pre_seconds: float = 3.0, post_seconds: float = 10.0,
                 device: str = 'auto'):
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.ocr_detector = ScoreOCRDetector(use_ocr=True, device=device)

    def detect_from_video(self, video_path: str, callback=None) -> List[dict]:
        """从视频中检测人头变化"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.ocr_detector.reset()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每 DETECT_INTERVAL_FRAMES 帧检测一次
            if frame_idx % DETECT_INTERVAL_FRAMES == 0:
                self.ocr_detector.analyze_frame(frame, frame_idx, fps)

            if callback:
                callback(frame_idx, total_frames)

            frame_idx += 1

        cap.release()

        events = self.ocr_detector.get_kill_events()

        highlights = []
        for event in events:
            start_frame = max(0, event.frame - int(self.pre_seconds * fps))
            end_frame = event.frame + int(self.post_seconds * fps)

            highlights.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_frame / fps,
                'end_time': end_frame / fps,
                'clip_type': 'kill',
                'confidence': event.confidence,
                'description': f"人头变化：{event.team} {event.old_score}→{event.new_score}"
            })

        return highlights
