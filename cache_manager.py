"""
视频分析结果缓存管理模块
缓存视频分析结果，避免重复分析相同视频
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CachedSegment:
    """缓存的片段数据"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    clip_type: str
    confidence: float
    description: str


@dataclass
class VideoCacheData:
    """视频缓存数据结构"""
    video_path: str
    video_hash: str
    file_size: int
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int
    analyze_time: str  # ISO 格式时间字符串
    segments: List[CachedSegment]
    config: Dict[str, Any]  # 分析时的配置参数


class VideoCacheManager:
    """视频缓存管理器"""
    
    def __init__(self, cache_dir: str = None):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录，默认为项目根目录下的 cache 文件夹
        """
        if cache_dir is None:
            self.cache_dir = Path(__file__).parent / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_video_hash(self, video_path: str) -> str:
        """计算视频文件的哈希值（使用文件路径 + 文件大小 + 修改时间）"""
        try:
            stat = os.stat(video_path)
            # 使用文件路径、大小和修改时间生成唯一标识
            hash_input = f"{video_path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        except Exception:
            return ""
    
    def _get_cache_file_path(self, video_path: str) -> Path:
        """获取缓存文件路径"""
        video_hash = self._compute_video_hash(video_path)
        # 使用哈希值作为缓存文件名
        return self.cache_dir / f"{video_hash}.json"
    
    def get_video_info(self, video_path: str) -> Dict:
        """获取视频基本信息（用于缓存）"""
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        cap.release()
        return info
    
    def save_cache(self, video_path: str, segments: List, 
                   config: Dict[str, Any], video_info: Dict = None) -> bool:
        """
        保存视频分析结果到缓存
        
        Args:
            video_path: 视频文件路径
            segments: 分析得到的片段列表
            config: 分析时的配置参数
            video_info: 视频基本信息（如果为 None 则自动获取）
        
        Returns:
            是否保存成功
        """
        try:
            if video_info is None:
                video_info = self.get_video_info(video_path)
            
            if not video_info:
                print(f"[CACHE] 无法获取视频信息：{video_path}")
                return False
            
            # 获取文件信息
            stat = os.stat(video_path)
            file_size = stat.st_size
            
            # 转换 segments 为可序列化格式
            cached_segments = []
            for seg in segments:
                cached_segments.append(CachedSegment(
                    start_frame=int(seg.start_frame),
                    end_frame=int(seg.end_frame),
                    start_time=float(seg.start_time),
                    end_time=float(seg.end_time),
                    clip_type=str(seg.clip_type),
                    confidence=float(seg.confidence),
                    description=str(seg.description)
                ))
            
            # 构建缓存数据
            cache_data = VideoCacheData(
                video_path=video_path,
                video_hash=self._compute_video_hash(video_path),
                file_size=file_size,
                duration=video_info.get('duration', 0),
                fps=video_info.get('fps', 0),
                width=video_info.get('width', 0),
                height=video_info.get('height', 0),
                total_frames=video_info.get('total_frames', 0),
                analyze_time=datetime.now().isoformat(),
                segments=cached_segments,
                config=config
            )
            
            # 保存到文件
            cache_file = self._get_cache_file_path(video_path)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(cache_data), f, ensure_ascii=False, indent=2)
            
            print(f"[CACHE] 已保存缓存：{cache_file}")
            return True
            
        except Exception as e:
            print(f"[CACHE] 保存缓存失败：{e}")
            return False
    
    def load_cache(self, video_path: str, 
                   check_config_match: bool = True,
                   current_config: Dict[str, Any] = None) -> Optional[VideoCacheData]:
        """
        加载视频分析结果缓存
        
        Args:
            video_path: 视频文件路径
            check_config_match: 是否检查配置参数匹配
            current_config: 当前配置参数（当 check_config_match=True 时使用）
        
        Returns:
            缓存数据，如果不存在或无效则返回 None
        """
        try:
            cache_file = self._get_cache_file_path(video_path)
            
            if not cache_file.exists():
                print(f"[CACHE] 未找到缓存：{cache_file}")
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证视频文件是否发生变化
            current_hash = self._compute_video_hash(video_path)
            if data.get('video_hash') != current_hash:
                print(f"[CACHE] 视频文件已变化，缓存失效")
                return None
            
            # 验证文件大小
            try:
                stat = os.stat(video_path)
                if data.get('file_size') != stat.st_size:
                    print(f"[CACHE] 文件大小已变化，缓存失效")
                    return None
            except Exception:
                pass
            
            # 检查配置参数是否匹配
            if check_config_match and current_config:
                cached_config = data.get('config', {})
                # 比较关键配置参数
                important_keys = ['confidence_threshold', 'use_ocr', 'detect_interval']
                for key in important_keys:
                    if key in current_config and key in cached_config:
                        if current_config[key] != cached_config[key]:
                            print(f"[CACHE] 配置参数 {key} 不匹配，缓存可能不准确")
                            # 配置不匹配时不直接返回 None，让用户决定
            
            # 转换回 VideoCacheData
            cached_segments = []
            for seg_data in data.get('segments', []):
                cached_segments.append(CachedSegment(**seg_data))
            
            cache_data = VideoCacheData(
                video_path=data.get('video_path', ''),
                video_hash=data.get('video_hash', ''),
                file_size=data.get('file_size', 0),
                duration=data.get('duration', 0),
                fps=data.get('fps', 0),
                width=data.get('width', 0),
                height=data.get('height', 0),
                total_frames=data.get('total_frames', 0),
                analyze_time=data.get('analyze_time', ''),
                segments=cached_segments,
                config=data.get('config', {})
            )
            
            print(f"[CACHE] 已加载缓存：{cache_file}")
            return cache_data
            
        except Exception as e:
            print(f"[CACHE] 加载缓存失败：{e}")
            return None
    
    def delete_cache(self, video_path: str) -> bool:
        """删除指定视频的缓存"""
        try:
            cache_file = self._get_cache_file_path(video_path)
            if cache_file.exists():
                cache_file.unlink()
                print(f"[CACHE] 已删除缓存：{cache_file}")
                return True
            return False
        except Exception as e:
            print(f"[CACHE] 删除缓存失败：{e}")
            return False
    
    def clear_all_cache(self) -> int:
        """清空所有缓存，返回删除的文件数量"""
        count = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                count += 1
            print(f"[CACHE] 已清空 {count} 个缓存文件")
        except Exception as e:
            print(f"[CACHE] 清空缓存失败：{e}")
        return count
    
    def get_cache_info(self, video_path: str) -> Optional[Dict]:
        """获取缓存信息（用于显示给用户）"""
        cache_data = self.load_cache(video_path, check_config_match=False)
        if cache_data is None:
            return None
        
        # 解析分析时间
        try:
            analyze_time = datetime.fromisoformat(cache_data.analyze_time)
            analyze_time_str = analyze_time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            analyze_time_str = cache_data.analyze_time
        
        return {
            'exists': True,
            'analyze_time': analyze_time_str,
            'segment_count': len(cache_data.segments),
            'duration': cache_data.duration,
            'config': cache_data.config
        }
