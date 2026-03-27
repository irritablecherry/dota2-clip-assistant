@echo off
chcp 65001 >nul
title Dota2 切片助手 - 打包工具

echo ╔========================================================╗
echo ║                Dota2 切片助手 - 打包工具                ║
echo ╚========================================================╝
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python 已安装

REM 运行打包脚本
echo.
echo 🚀 开始打包...
echo.
python build.py

echo.
echo 打包完成！
pause
