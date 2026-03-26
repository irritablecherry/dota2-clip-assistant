"""
比分区域图片 OCR 识别
对 images 文件夹中的图片进行比分区域 OCR 识别
比分区域内：左侧 25% 为天灾比分，右侧 25% 为夜魇比分，中间 50% 为分隔区域
使用 YOLO 检测比分区域，然后进行 OCR 识别
"""
import cv2
from pathlib import Path
import easyocr
import numpy as np
from ultralytics import YOLO


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


def recognize_score_in_area(score_area: np.ndarray, reader: easyocr.Reader) -> tuple:
    """
    在比分区域内识别左右两侧比分和中间时间
    
    比分区域划分：
    - 左侧 25%：天灾比分
    - 中间 50%：分隔区域（包含时间）
    - 右侧 25%：夜魇比分
    """
    if score_area is None or score_area.size == 0:
        return None, None, None
    
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
    time_str = recognize_time_in_region(middle_region, reader)
    
    return left_digit, right_digit, time_str


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
        print(f"  {region_name}区域识别失败：{e}")
        return None


def recognize_time_in_region(region: np.ndarray, reader: easyocr.Reader) -> str:
    """识别中间区域的时间（格式如 00:00）"""
    try:
        # 使用允许列表识别数字和冒号
        results = reader.readtext(region, paragraph=False, allowlist='0123456789:')
        
        if not results:
            return None

        print(f'results:{results}')
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
            if text and ":" not in text and len(text)>=3:
                index = len(text) - 2
                cleaned = text[:index] + ":" + text[index:]
                if cleaned:
                    texts.append({
                        'text': cleaned,
                        'confidence': conf
                    })
        if not texts:
            return None
        
        # 返回置信度最高的结果
        best_text = max(texts, key=lambda t: t['confidence'])
        return best_text['text']
        
    except Exception as e:
        print(f"  时间区域识别失败：{e}")
        return None


def analyze_images():
    """分析 images 文件夹中的所有图片"""
    base_dir = Path(__file__).parent
    images_dir = base_dir / "images"
    
    print("=" * 70)
    print("🎮 比分区域图片 OCR 识别")
    print("=" * 70)
    
    # 获取图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(ext))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print("\n❌ images 文件夹中没有图片!")
        return
    
    print(f"\n📁 找到 {len(image_files)} 张图片:")
    for img_file in image_files:
        print(f"   - {img_file.name}")
    
    # 初始化 EasyOCR
    print(f"\n🔧 初始化 EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    
    # 加载 YOLO 模型
    model_path = base_dir / "model" / "best.pt"
    print(f"🔧 加载 YOLO 模型：{model_path.name}...")
    model = YOLO(str(model_path))
    
    # 统计信息
    total_images = len(image_files)
    success_count = 0
    left_only_count = 0
    right_only_count = 0
    both_count = 0
    failed_count = 0
    
    results = []
    
    print(f"\n{'=' * 70}")
    print("📊 开始识别...")
    print(f"{'=' * 70}\n")
    
    for img_path in image_files:
        short_path = get_short_path_name(str(img_path))
        if short_path:
            img_path_for_cv = Path(short_path)
        else:
            img_path_for_cv = img_path
        
        # 读取图片
        img = cv2.imread(str(img_path_for_cv))
        if img is None:
            print(f"❌ 无法读取图片：{img_path.name}")
            failed_count += 1
            continue
        
        print(f"📄 处理：{img_path.name}")
        print(f"   尺寸：{img.shape[1]} x {img.shape[0]}")
        
        # YOLO 检测比分区域
        yolo_results = model(img, conf=0.3, verbose=False)
        score_area = None
        
        if yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                class_name = model.names[cls_id]
                if class_name == 'score_area':
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    score_area = img[y1:y2, x1:x2]
                    print(f"   YOLO 检测到比分区域：x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")
                    break
        
        if score_area is None:
            print(f"   ⚠️  未检测到比分区域，使用默认 ROI")
            # 使用默认 ROI 作为备选
            h, w = img.shape[:2]
            roi = (
                int(w * 0.35),   # x
                int(h * 0.02),   # y
                int(w * 0.30),   # width
                int(h * 0.08)    # height
            )
            x, y, roi_w, roi_h = roi
            score_area = img[y:y+roi_h, x:x+roi_w]
        
        # 识别比分和时间
        left_digit, right_digit, time_str = recognize_score_in_area(score_area, reader)
        
        # 输出结果
        time_display = time_str if time_str else "??:??"
        if left_digit is not None and right_digit is not None:
            print(f"   ✅ 识别结果：{left_digit} : {right_digit}  时间：{time_display}")
            both_count += 1
            success_count += 1
            results.append({
                'file': img_path.name,
                'left': left_digit,
                'right': right_digit,
                'time': time_str,
                'status': 'success'
            })
        elif left_digit is not None and right_digit is None:
            print(f"   ⚠️  仅识别到左侧：{left_digit} : ?  时间：{time_display}")
            left_only_count += 1
            success_count += 1
            results.append({
                'file': img_path.name,
                'left': left_digit,
                'right': None,
                'time': time_str,
                'status': 'left_only'
            })
        elif left_digit is None and right_digit is not None:
            print(f"   ⚠️  仅识别到右侧：? : {right_digit}  时间：{time_display}")
            right_only_count += 1
            success_count += 1
            results.append({
                'file': img_path.name,
                'left': None,
                'right': right_digit,
                'time': time_str,
                'status': 'right_only'
            })
        else:
            print(f"   ❌ 识别失败  时间：{time_display}")
            failed_count += 1
            results.append({
                'file': img_path.name,
                'left': None,
                'right': None,
                'time': time_str,
                'status': 'failed'
            })
        print()
    
    # 统计结果
    print(f"\n{'=' * 70}")
    print("📈 识别结果统计")
    print(f"{'=' * 70}\n")
    
    print(f"  总图片数：{total_images}")
    print(f"  成功识别：{success_count} ({success_count/total_images*100:.1f}%)")
    print(f"  识别失败：{failed_count} ({failed_count/total_images*100:.1f}%)\n")
    
    print(f"  识别情况:")
    if success_count > 0:
        print(f"  - 左右都识别到：{both_count} ({both_count/success_count*100:.1f}%)")
        print(f"  - 仅识别到左侧：{left_only_count} ({left_only_count/success_count*100:.1f}%)")
        print(f"  - 仅识别到右侧：{right_only_count} ({right_only_count/success_count*100:.1f}%)\n")
    
    print(f"  详细结果:")
    for r in results:
        time_display = r['time'] if r['time'] else "??:??"
        if r['status'] == 'success':
            print(f"    - {r['file']}: {r['left']} : {r['right']}  时间：{time_display} ✅")
        elif r['status'] == 'left_only':
            print(f"    - {r['file']}: {r['left']} : ?  时间：{time_display} ⚠️")
        elif r['status'] == 'right_only':
            print(f"    - {r['file']}: ? : {r['right']}  时间：{time_display} ⚠️")
        else:
            print(f"    - {r['file']}: 识别失败  时间：{time_display} ❌")
    
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    analyze_images()
