"""
视频比分区域 OCR 识别
对 videos 文件夹中的视频进行比分区域 OCR 识别
使用 YOLO 检测比分区域，然后进行 OCR 识别
比分区域内：左侧 25% 为天灾比分，右侧 25% 为夜魇比分，中间 50% 为分隔区域（包含时间）
"""
import cv2
from pathlib import Path
import easyocr
import numpy as np
from ultralytics import YOLO
import subprocess
import time

# 检测间隔帧数
DETECT_INTERVAL_FRAMES = 30


def get_short_path_name(long_path):
    """获取 Windows 短路径名"""
    try:
        import ctypes
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        GetShortPathNameW.restype = ctypes.c_uint

        output_buf_size = 0
        while True:
            output_buf = ctypes.create_unicode_buffer(output_buf_size)
            size = GetShortPathNameW(long_path, output_buf, output_buf_size)
            if size == 0:
                return None
            if size <= output_buf_size:
                return output_buf.value
            output_buf_size = size
    except Exception:
        return None


def recognize_single_region(region: np.ndarray, reader: easyocr.Reader, region_name: str) -> int:
    """识别单个区域（左侧或右侧）的数字"""
    try:
        results = reader.readtext(region, paragraph=False, allowlist='0123456789')
        
        if not results:
            return None
        
        # 提取数字
        digits = []
        for (bbox, text, conf) in results:
            if text and text.isdigit():
                digits.append({
                    'text': int(text),
                    'confidence': conf
                })
        
        if not digits:
            return None
        
        # 返回置信度最高的数字
        best_digit = max(digits, key=lambda d: d['confidence'])
        return best_digit['text']
        
    except Exception as e:
        return None


def recognize_time_in_region(region: np.ndarray, reader: easyocr.Reader) -> tuple:
    """
    识别中间区域的时间（格式如 00:00）
    返回：(时间字符串，置信度)
    """
    try:
        # 使用允许列表识别数字和冒号
        results = reader.readtext(region, paragraph=False, allowlist='0123456789:')

        if not results:
            return None, 0.0
        # print(f'results:{results}')
        # 提取识别到的文本
        texts = []
        for (bbox, text, conf) in results:
            if text and ":" in text:
                # 清理文本，保留数字和冒号
                cleaned = ''.join(c for c in text if c.isdigit() or c == ':')
                if cleaned:
                    texts.append({
                        'text': cleaned,
                        'confidence': conf
                    })
            if text and ":" not in text and len(text) >= 3:
                index = len(text) - 2
                cleaned = text[:index] + ":" + text[index:]
                if cleaned:
                    texts.append({
                        'text': cleaned,
                        'confidence': conf
                    })

        if not texts:
            return None, 0.0

        # 返回置信度最高的结果
        best_text = max(texts, key=lambda t: t['confidence'])
        return best_text['text'], best_text['confidence']

    except Exception as e:
        return None, 0.0


def recognize_score_in_area(score_area: np.ndarray, reader: easyocr.Reader) -> tuple:
    """
    在比分区域内识别左右两侧比分和中间时间

    比分区域划分：
    - 左侧 25%：天灾比分
    - 中间 50%：分隔区域（包含时间）
    - 右侧 25%：夜魇比分
    
    返回：(左侧比分，右侧比分，时间字符串，时间置信度)
    """
    if score_area is None or score_area.size == 0:
        return None, None, None, 0.0

    # 放大 4 倍
    score_area_scaled = cv2.resize(score_area, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    img_height, img_width = score_area_scaled.shape[:2]

    # 计算区域边界（左侧 25%，中间 50%，右侧 25%）
    left_region_width = int(img_width * 0.25)
    middle_region_start = left_region_width
    middle_region_end = int(img_width * 0.75)
    right_region_start = middle_region_end

    # 提取三个区域
    left_region = score_area_scaled[:, :left_region_width]
    middle_region = score_area_scaled[:, middle_region_start:middle_region_end]
    right_region = score_area_scaled[:, right_region_start:]

    # 分别识别
    left_digit = recognize_single_region(left_region, reader, "左侧")
    right_digit = recognize_single_region(right_region, reader, "右侧")
    time_str, time_conf = recognize_time_in_region(middle_region, reader)

    return left_digit, right_digit, time_str, time_conf


def analyze_video():
    """分析 videos 文件夹中的视频"""
    base_dir = Path(__file__).parent
    video_dir = base_dir / "videos"
    model_path = base_dir / "model" / "best.pt"
    
    print("=" * 70)
    print("🎮 视频比分区域 OCR 识别")
    print("=" * 70)
    
    # 获取视频文件
    result = subprocess.run(
        ["dir", "/b", str(video_dir)],
        shell=True,
        capture_output=True,
        text=True,
        encoding='gbk'
    )
    video_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
    
    if not video_files:
        print("\n❌ videos 文件夹为空!")
        return
    
    video_file = video_files[0]
    video_path = video_dir / video_file
    short_path = get_short_path_name(str(video_path))
    if short_path:
        video_path = Path(short_path)
    
    print(f"\n📁 测试视频：{video_file}")
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"   帧率：{fps:.2f} FPS")
    print(f"   总帧数：{total_frames}")
    print(f"   时长：{duration:.2f} 秒")
    
    # 加载 YOLO 模型
    print(f"\n🔧 加载 YOLO 模型...")
    model = YOLO(str(model_path))
    
    # 初始化 EasyOCR
    print("🔧 初始化 EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    
    # 开始测试
    print(f"\n{'=' * 70}")
    print(f"📊 开始识别（每 {DETECT_INTERVAL_FRAMES} 帧检测一次）...")
    print(f"{'=' * 70}\n")
    
    # 统计信息
    total_detected = 0
    left_only_count = 0
    right_only_count = 0
    both_count = 0
    failed_count = 0
    all_scores = []

    # 缓存上次识别结果，初始为 0:0
    last_left = 0
    last_right = 0
    
    # 时间缓存和校验
    last_time_str = "0:00"
    last_time_conf = 0.0
    TIME_CONFIDENCE_THRESHOLD = 0.9
    time_history = []  # 存储 (frame_idx, time_str, conf) 用于校验
    
    # 时间变化趋势检测
    time_trend = None  # None, "increasing", "decreasing"
    last_valid_sec = None

    start_time = time.perf_counter()
    frame_idx = 0

    def parse_time_to_seconds(time_str):
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

    def seconds_to_time_str(seconds):
        """将秒数转换为时间字符串（格式：1:03 而不是 01:03）"""
        if seconds is None:
            return "??:??"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def detect_time_trend(time_history):
        """根据历史数据检测时间变化趋势"""
        if len(time_history) < 2:
            return None
        
        # 使用最近 2 次高置信度记录判断趋势
        recent = [h for h in time_history[-5:] if h[2] >= TIME_CONFIDENCE_THRESHOLD]
        if len(recent) < 2:
            return time_trend  # 保持原有趋势
        
        # 比较首尾时间变化
        first_sec = parse_time_to_seconds(recent[0][1])
        last_sec = parse_time_to_seconds(recent[-1][1])
        
        if first_sec is not None and last_sec is not None:
            if last_sec > first_sec:
                return "increasing"
            elif last_sec < first_sec:
                return "decreasing"
        return time_trend

    def get_expected_time_range(time_history, trend, fps):
        """
        根据历史数据和趋势，返回预期的时间范围
        返回：(min_sec, max_sec) 或 None
        """
        if not time_history:
            return None
        
        # 获取最近的高置信度时间
        recent = [h for h in time_history[-5:] if h[2] >= TIME_CONFIDENCE_THRESHOLD]
        if not recent:
            return None
        
        last_frame, last_time_str, _ = recent[-1]
        last_sec = parse_time_to_seconds(last_time_str)
        
        if last_sec is None:
            return None
        
        # 计算帧数差
        frames_elapsed = frame_idx - last_frame
        if frames_elapsed <= 0:
            return (last_sec, last_sec)
        
        # 根据趋势计算预期范围
        # 假设最快速度：1 秒/帧（极端情况），最慢：0 秒/帧（暂停）
        max_change = frames_elapsed / fps  # 最大可能变化秒数
        
        if trend == "increasing":
            return (last_sec, last_sec + max_change * 1.5)  # 允许一定超调
        elif trend == "decreasing":
            return (last_sec - max_change * 1.5, last_sec)
        else:
            # 趋势不明，允许双向变化
            return (last_sec - max_change, last_sec + max_change)

    def is_time_reasonable(time_str, expected_range):
        """检查时间是否在预期范围内"""
        if expected_range is None:
            return True  # 无法计算预期范围时，认为都合理
        
        t_sec = parse_time_to_seconds(time_str)
        if t_sec is None:
            return False
        
        min_sec, max_sec = expected_range
        # 允许额外 3 秒容差
        return (min_sec - 3) <= t_sec <= (max_sec + 3)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每 DETECT_INTERVAL_FRAMES 帧检测一次
        if frame_idx % DETECT_INTERVAL_FRAMES == 0:
            # YOLO 检测比分区域
            yolo_results = model(frame, conf=0.3, verbose=False)
            score_area = None

            if yolo_results[0].boxes is not None:
                boxes = yolo_results[0].boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    class_name = model.names[cls_id]
                    if class_name == 'score_area':
                        bbox = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, bbox)
                        score_area = frame[y1:y2, x1:x2]
                        break

            if score_area is not None:
                total_detected += 1

                # 识别比分和时间
                left_digit, right_digit, time_str, time_conf = recognize_score_in_area(score_area, reader)

                # 如果没有识别到比分，使用上次结果
                if left_digit is None:
                    left_digit = last_left
                if right_digit is None:
                    right_digit = last_right

                # 更新比分缓存
                last_left = left_digit
                last_right = right_digit

                # 时间处理逻辑
                time_display = "??:??"
                time_conf_display = 0.0
                
                # 检测时间变化趋势
                time_trend = detect_time_trend(time_history)
                
                # 获取预期时间范围
                expected_range = get_expected_time_range(time_history, time_trend, fps)
                
                # 情况 1: 识别失败或置信度低，使用预期时间或上次结果
                if time_str is None or time_conf < TIME_CONFIDENCE_THRESHOLD:
                    if expected_range is not None:
                        # 使用预期范围的中间值
                        min_sec, max_sec = expected_range
                        expected_sec = (min_sec + max_sec) / 2
                        time_display = seconds_to_time_str(expected_sec)
                        time_conf_display = last_time_conf
                    else:
                        time_display = last_time_str
                        time_conf_display = last_time_conf
                # 情况 2: 高置信度但时间明显异常（超出预期范围）
                elif not is_time_reasonable(time_str, expected_range):
                    time_display = last_time_str
                    time_conf_display = last_time_conf
                    trend_info = f" (趋势：{time_trend})" if time_trend else ""
                    print(f"  [时间修正] 帧 {frame_idx}: 识别到 '{time_str}' ({time_conf:.2f}), "
                          f"超出预期范围，使用上次值 '{last_time_str}'{trend_info}")
                # 情况 3: 高置信度且时间合理，使用识别到的时间
                else:
                    time_display = time_str
                    time_conf_display = time_conf
                    # 更新缓存和历史
                    last_time_str = time_str
                    last_time_conf = time_conf
                    time_history.append((frame_idx, time_str, time_conf))

                # 输出结果
                if left_digit is not None and right_digit is not None:
                    timestamp = frame_idx / fps
                    print(f"帧 {frame_idx} ({timestamp:.1f}s): {left_digit} : {right_digit}  时间：{time_display} ✅")
                    both_count += 1
                    all_scores.append((left_digit, right_digit))
                elif left_digit is not None and right_digit is None:
                    timestamp = frame_idx / fps
                    print(f"帧 {frame_idx} ({timestamp:.1f}s): {left_digit} : ?  时间：{time_display} ⚠️")
                    left_only_count += 1
                elif left_digit is None and right_digit is not None:
                    timestamp = frame_idx / fps
                    print(f"帧 {frame_idx} ({timestamp:.1f}s): ? : {right_digit}  时间：{time_display} ⚠️")
                    right_only_count += 1
                else:
                    timestamp = frame_idx / fps
                    print(f"帧 {frame_idx} ({timestamp:.1f}s): 识别失败  时间：{time_display} ❌")
                    failed_count += 1

        frame_idx += 1
    
    total_time = time.perf_counter() - start_time
    cap.release()
    
    # 统计结果
    print(f"\n{'=' * 70}")
    print("📈 识别结果统计")
    print(f"{'=' * 70}\n")
    
    detected_frames = total_detected
    
    print(f"  处理统计:")
    print(f"  - 处理帧数：{frame_idx}")
    print(f"  - 检测帧数（每 {DETECT_INTERVAL_FRAMES} 帧）: {detected_frames}")
    print(f"  - 总用时：{total_time:.2f} 秒")
    print(f"  - 平均帧率：{frame_idx/total_time:.2f} FPS\n")
    
    print(f"  检测结果:")
    if detected_frames > 0:
        print(f"  - 检测到比分区域：{detected_frames}")
        print(f"  - 左右都识别到：{both_count} ({both_count/detected_frames*100:.1f}%)")
        print(f"  - 仅识别到左侧：{left_only_count} ({left_only_count/detected_frames*100:.1f}%)")
        print(f"  - 仅识别到右侧：{right_only_count} ({right_only_count/detected_frames*100:.1f}%)")
        print(f"  - 识别失败：{failed_count} ({failed_count/detected_frames*100:.1f}%)\n")
    
    print(f"  所有检测到的比分:")
    unique_scores = list(set(all_scores))
    for score in sorted(unique_scores):
        count = all_scores.count(score)
        print(f"    - {score[0]} : {score[1]} ({count} 次)")
    
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    analyze_video()
