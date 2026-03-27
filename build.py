"""
Dota2 Clip Assistant - 打包编译脚本
使用 PyInstaller 将程序打包为 exe 可执行文件
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent
BUILD_DIR = BASE_DIR / "build"
DIST_DIR = BASE_DIR / "dist"
SPEC_FILE = BASE_DIR / "dota2_clip_assistant.spec"

# 需要打包的文件
DATA_FILES = [
    ("model", "model"),           # YOLO 模型
    ("config.json", "config.json"),  # 配置文件
]

# 需要排除的文件夹
EXCLUDE_DIRS = [
    "__pycache__",
    ".git",
    ".idea",
    ".venv",
    "build",
    "dist",
    "cache",
    "videos",
    "images",
    "outputs",
]


def check_requirements():
    """检查是否安装了必要的依赖"""
    print("=" * 60)
    print("🔍 检查依赖项...")
    print("=" * 60)
    
    try:
        import PyInstaller
        print(f"✅ PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("❌ PyInstaller 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✅ PyInstaller 安装完成")
    
    try:
        import av
        print(f"✅ PyAV: {av.__version__}")
    except ImportError:
        print("❌ PyAV 未安装")
        return False
    
    try:
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics 未安装")
        return False
    
    try:
        import easyocr
        print(f"✅ EasyOCR: 已安装")
    except ImportError:
        print("❌ EasyOCR 未安装")
        return False
    
    try:
        from PyQt5 import QtCore
        print(f"✅ PyQt5: {QtCore.PYQT_VERSION_STR}")
    except ImportError:
        print("❌ PyQt5 未安装")
        return False
    
    print()
    return True


def clean_build():
    """清理旧的构建文件"""
    print("=" * 60)
    print("🧹 清理旧的构建文件...")
    print("=" * 60)
    
    # 删除 build 目录
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print(f"📁 已删除：{BUILD_DIR}")
    
    # 删除 dist 目录
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
        print(f"📁 已删除：{DIST_DIR}")
    
    # 删除 spec 文件
    if SPEC_FILE.exists():
        SPEC_FILE.unlink()
        print(f"📄 已删除：{SPEC_FILE}")
    
    print()


def build_exe(onefile=False):
    """
    构建 exe 文件
    
    Args:
        onefile: True=单个 exe 文件，False=目录模式
    """
    print("=" * 60)
    print("🚀 开始打包...")
    print("=" * 60)
    
    # 构建 PyInstaller 命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "dota2_clip_assistant",
        "--windowed",  # 不显示控制台
        "--noconfirm",  # 覆盖时不询问
    ]
    
    # 单文件模式添加 --onefile
    if onefile:
        cmd.append("--onefile")
        print("📦 打包模式：单文件 exe")
    else:
        print("📦 打包模式：目录模式")
    
    # 添加数据文件
    for src, dst in DATA_FILES:
        src_path = BASE_DIR / src
        if src_path.exists():
            cmd.append("--add-data")
            if os.name == 'nt':  # Windows
                cmd.append(f"{src_path};{dst}")
            else:  # Linux/Mac
                cmd.append(f"{src_path}:{dst}")
            print(f"📎 添加数据：{src} → {dst}")
    
    # 排除不必要的模块（减小体积）
    exclude_modules = [
        "matplotlib",
        "scipy",
        "sklearn",
        "tensorflow",
        "keras",
        "jupyter",
        "notebook",
        "IPython",
    ]
    for module in exclude_modules:
        cmd.append("--exclude-module")
        cmd.append(module)
    
    # 添加隐藏导入（确保必要的模块被包含）
    hidden_imports = [
        "ultralytics",
        "av",
        "easyocr",
        "PyQt5",
        "cv2",
        "numpy",
        "torch",
        "torchvision",
    ]
    for module in hidden_imports:
        cmd.append("--hidden-import")
        cmd.append(module)
    
    # 添加图标（使用 logo.ico）
    icon_path = BASE_DIR / "logo.ico"
    if icon_path.exists():
        cmd.append("--icon")
        cmd.append(str(icon_path))
        print(f"🎨 添加图标：{icon_path}")
    else:
        # 尝试 icon.ico
        icon_path = BASE_DIR / "icon.ico"
        if icon_path.exists():
            cmd.append("--icon")
            cmd.append(str(icon_path))
            print(f"🎨 添加图标：{icon_path}")
    
    # 添加主程序入口
    cmd.append(str(BASE_DIR / "main_window.py"))
    
    print()
    print("📋 执行命令:")
    print(" ".join(cmd))
    print()
    
    # 执行打包
    try:
        subprocess.run(cmd, check=True, cwd=str(BASE_DIR))
        print()
        print("=" * 60)
        print("✅ 打包完成!")
        print("=" * 60)
        
        # 显示输出位置
        if onefile:
            exe_path = DIST_DIR / "dota2_clip_assistant.exe"
        else:
            exe_path = DIST_DIR / "dota2_clip_assistant" / "dota2_clip_assistant.exe"
        
        print(f"📦 可执行文件：{exe_path}")
        print(f"📁 输出目录：{DIST_DIR}")
        
        # 复制额外文件到输出目录
        copy_extra_files(DIST_DIR, onefile)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"❌ 打包失败：{e}")
        print("=" * 60)
        return False


def copy_extra_files(dist_dir, onefile):
    """复制额外文件到输出目录"""
    print()
    print("📋 复制额外文件...")
    
    if onefile:
        target_dir = dist_dir
    else:
        target_dir = dist_dir / "dota2_clip_assistant"
    
    # 复制 README
    readme_path = BASE_DIR / "README.md"
    if readme_path.exists():
        shutil.copy2(readme_path, target_dir / "README.md")
        print(f"📄 已复制：README.md")
    
    # 复制 logo.ico（用于快捷方式图标）
    logo_path = BASE_DIR / "logo.ico"
    if logo_path.exists():
        shutil.copy2(logo_path, target_dir / "logo.ico")
        print(f"🎨 已复制：logo.ico")
    
    # 创建使用说明文件
    readme_txt = target_dir / "使用说明.txt"
    with open(readme_txt, "w", encoding="utf-8") as f:
        f.write("Dota2 切片助手 - 使用说明\n")
        f.write("=" * 50 + "\n\n")
        f.write("1. 首次运行会自动下载必要的模型文件\n")
        f.write("2. 建议将程序放在 SSD 硬盘上以获得更好的性能\n")
        f.write("3. 如需使用 FFmpeg 加速，请确保已安装 FFmpeg\n")
        f.write("   下载地址：https://ffmpeg.org/download.html\n\n")
        f.write("运行方式:\n")
        f.write("- 双击 dota2_clip_assistant.exe 启动程序\n")
        f.write("- 选择 Dota2 比赛视频文件\n")
        f.write("- 点击'分析视频'按钮\n")
        f.write("- 分析完成后点击'提取片段'按钮\n\n")
        f.write("注意事项:\n")
        f.write("- 首次运行需要下载模型文件（约 50MB）\n")
        f.write("- 分析速度取决于 CPU/GPU 性能\n")
        f.write("- 使用 FFmpeg 可大幅提升视频处理速度\n")
    print(f"📄 已创建：使用说明.txt")


def create_shortcut():
    """创建桌面快捷方式（仅 Windows）"""
    if os.name != 'nt':
        return
    
    try:
        import win32com.client
        
        dist_dir = DIST_DIR / "dota2_clip_assistant"
        exe_path = dist_dir / "dota2_clip_assistant.exe"
        
        if not exe_path.exists():
            print("⚠️ 未找到 exe 文件，跳过快捷方式创建")
            return
        
        # 桌面路径
        desktop = Path.home() / "Desktop"
        shortcut_path = desktop / "Dota2 切片助手.lnk"
        
        # 图标路径（优先使用 logo.ico）
        icon_path = BASE_DIR / "logo.ico"
        if not icon_path.exists():
            icon_path = BASE_DIR / "icon.ico"
        
        # 如果 dist 目录有图标文件，使用 dist 目录的
        dist_icon = dist_dir / "logo.ico"
        if dist_icon.exists():
            icon_path = dist_icon
        elif not icon_path.exists():
            # 使用 exe 内置图标
            icon_path = exe_path
        
        # 创建快捷方式
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(shortcut_path))
        shortcut.TargetPath = str(exe_path)
        shortcut.WorkingDirectory = str(dist_dir)
        shortcut.IconLocation = str(icon_path)
        shortcut.save()
        
        print(f"🔗 已创建桌面快捷方式：{shortcut_path}")
        print(f"🎨 快捷方式图标：{icon_path}")
        
    except Exception as e:
        print(f"⚠️ 创建快捷方式失败：{e}")


def main():
    """主函数"""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "Dota2 切片助手 - 打包工具" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 检查依赖
    if not check_requirements():
        print("\n❌ 依赖检查失败，请先安装必要的依赖")
        print("运行：pip install -r requirements.txt")
        input("\n按回车键退出...")
        return
    
    # 清理旧构建
    clean_build()
    
    # 选择打包模式
    print("请选择打包模式:")
    print("1. 单文件 exe (体积较大，启动较慢，便于分发)")
    print("2. 目录模式 (体积较小，启动较快，文件较多)")
    print()
    
    choice = input("请输入选项 (1/2)，默认为 2: ").strip()
    onefile = (choice == "1")
    
    print()
    
    # 开始打包
    success = build_exe(onefile=onefile)
    
    if success:
        print()
        print("=" * 60)
        print("🎉 打包成功完成!")
        print("=" * 60)
        
        # 询问是否创建快捷方式
        if os.name == 'nt':
            create_shortcut_input = input("\n是否创建桌面快捷方式？(y/n): ").strip().lower()
            if create_shortcut_input == 'y':
                create_shortcut()
    else:
        print()
        print("=" * 60)
        print("💥 打包失败，请检查错误信息")
        print("=" * 60)
    
    print()
    input("按回车键退出...")


if __name__ == "__main__":
    main()
